"""Baseline thực hành pruning cho SSD MobileNet của Lab 2/3.

Script này minh họa hai dạng pruning bằng PyTorch:

* unstructured: đặt các trọng số nhỏ nhất về 0 trên toàn bộ Conv2d/Linear;
* structured: đặt cả filter (output channel) có chuẩn nhỏ về 0 trên Conv2d.

`torch.nn.utils.prune` tạo mask nhưng không tự thay đổi topology của mạng.
Vì vậy, structured pruning trong script này là structured *sparsity* để đo
tác động đến chất lượng; muốn giảm FLOPs/latency thực sự cần bước compact
model hoặc runtime có sparse kernel.

Ví dụ:
    python pruning_baseline.py --checkpoint runs/best.pt \
        --method unstructured --amount 0.5 --out results/prune_unstructured.pt

    python pruning_baseline.py --checkpoint runs/best.pt \
        --method structured --amount 0.3 --out results/prune_structured.pt
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn.utils import prune
from torchvision.models.detection import ssdlite320_mobilenet_v3_large

try:
    import psutil
except ImportError:  # pragma: no cover - psutil là dependency tùy chọn của lab
    psutil = None


def build_model(num_classes: int) -> nn.Module:
    """Tạo đúng kiến trúc SSD MobileNet dùng trong Lab 2."""
    return ssdlite320_mobilenet_v3_large(
        weights=None,
        weights_backbone=None,
        num_classes=num_classes,
    )


def load_checkpoint(model: nn.Module, checkpoint: str) -> None:
    """Nạp checkpoint dạng state_dict hoặc dict có khóa model/state_dict."""
    payload: Any = torch.load(checkpoint, map_location="cpu")
    if isinstance(payload, dict) and "model" in payload:
        state = payload["model"]
    elif isinstance(payload, dict) and "state_dict" in payload:
        state = payload["state_dict"]
    else:
        state = payload

    if not isinstance(state, dict):
        raise TypeError("Checkpoint phải là state_dict hoặc dict có khóa 'model'.")

    cleaned = {
        key.removeprefix("module."): value
        for key, value in state.items()
    }
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            "Checkpoint không khớp kiến trúc SSD MobileNet. "
            f"missing={missing[:5]}, unexpected={unexpected[:5]}"
        )


def prunable_modules(model: nn.Module, structured: bool) -> list[nn.Module]:
    """Chọn các layer có weight để áp dụng pruning."""
    allowed = (nn.Conv2d,) if structured else (nn.Conv2d, nn.Linear)
    return [
        module
        for module in model.modules()
        if isinstance(module, allowed) and module.weight.numel() > 0
    ]


def apply_pruning(model: nn.Module, method: str, amount: float) -> dict[str, int]:
    """Áp dụng pruning rồi remove re-parametrization để model có weight thật."""
    structured = method == "structured"
    modules = prunable_modules(model, structured=structured)
    if not modules:
        raise RuntimeError("Không tìm thấy Conv2d/Linear để pruning.")

    if method == "unstructured":
        parameters = [(module, "weight") for module in modules]
        prune.global_unstructured(
            parameters,
            pruning_method=prune.L1Unstructured,
            amount=amount,
        )
    elif method == "structured":
        for module in modules:
            prune.ln_structured(
                module,
                name="weight",
                amount=amount,
                n=2,
                dim=0,
            )
    else:
        raise ValueError("method phải là 'unstructured' hoặc 'structured'.")

    # Xóa weight_orig/weight_mask để checkpoint cuối có cấu trúc state_dict
    # bình thường; các phần tử đã bị mask vẫn bằng 0.
    for module in modules:
        prune.remove(module, "weight")

    return {
        "layers": len(modules),
        "prunable_parameters": sum(module.weight.numel() for module in modules),
    }


def percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    index = (len(ordered) - 1) * q
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    fraction = index - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def rss_mb() -> float | None:
    if psutil is None:
        return None
    return psutil.Process().memory_info().rss / (1024 * 1024)


@torch.no_grad()
def benchmark(
    model: nn.Module,
    device: torch.device,
    input_size: int,
    warmup: int,
    iterations: int,
) -> dict[str, float | int | None]:
    """Đo latency inference của detector với cùng một input cố định."""
    model.eval().to(device)
    image = torch.zeros(3, input_size, input_size, device=device)

    for _ in range(warmup):
        model([image])
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    start_rss = rss_mb()
    process = psutil.Process() if psutil is not None else None
    if process is not None:
        process.cpu_percent(None)
        psutil.cpu_percent(None)
    timings: list[float] = []
    for _ in range(iterations):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        model([image])
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        timings.append((time.perf_counter() - start) * 1000.0)

    mean_ms = sum(timings) / len(timings)
    return {
        "warmup": warmup,
        "iterations": iterations,
        "mean_ms": mean_ms,
        "std_ms": math.sqrt(
            sum((value - mean_ms) ** 2 for value in timings) / len(timings)
        ),
        "p50_ms": percentile(timings, 0.50),
        "p95_ms": percentile(timings, 0.95),
        "fps": 1000.0 / mean_ms if mean_ms > 0 else float("inf"),
        "rss_before_mb": start_rss,
        "rss_after_mb": rss_mb(),
        "process_cpu_percent": process.cpu_percent(None) if process is not None else None,
        "system_cpu_percent": psutil.cpu_percent(None) if psutil is not None else None,
    }


def parameter_report(model: nn.Module) -> dict[str, int | float]:
    """Tính sparsity trên toàn bộ parameter và riêng các weight có thể prune."""
    all_parameters = [parameter.detach() for parameter in model.parameters()]
    weights = [
        module.weight.detach()
        for module in model.modules()
        if isinstance(module, (nn.Conv2d, nn.Linear))
    ]

    def summarize(values: list[torch.Tensor]) -> dict[str, int | float]:
        total = sum(value.numel() for value in values)
        zeros = sum(int((value == 0).sum().item()) for value in values)
        return {
            "parameters": total,
            "zero_parameters": zeros,
            "nonzero_parameters": total - zeros,
            "sparsity": zeros / total if total else 0.0,
        }

    filters = 0
    zero_filters = 0
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            flattened = module.weight.detach().reshape(module.out_channels, -1)
            filters += module.out_channels
            zero_filters += int((flattened.abs().sum(dim=1) == 0).sum().item())

    result: dict[str, int | float] = {
        "total_parameters": sum(value.numel() for value in all_parameters),
        "zero_parameters": sum(int((value == 0).sum().item()) for value in all_parameters),
        "filter_count": filters,
        "zero_filters": zero_filters,
        "filter_sparsity": zero_filters / filters if filters else 0.0,
    }
    result["overall_sparsity"] = (
        result["zero_parameters"] / result["total_parameters"]
        if result["total_parameters"]
        else 0.0
    )
    result["prunable_sparsity"] = summarize(weights)["sparsity"]
    return result


def save_model(model: nn.Module, path: Path) -> float:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict()}, path)
    return path.stat().st_size / (1024 * 1024)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="runs/best.pt của Lab 2")
    parser.add_argument("--method", choices=("unstructured", "structured"), required=True)
    parser.add_argument("--amount", type=float, required=True, help="Tỷ lệ prune trong [0, 1)")
    parser.add_argument("--out", required=True, help="Checkpoint model sau pruning")
    parser.add_argument("--report", help="JSON kết quả; mặc định cùng thư mục với --out")
    parser.add_argument("--num-classes", type=int, default=21)
    parser.add_argument("--input-size", type=int, default=320)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--threads", type=int, help="Số CPU threads của PyTorch")
    parser.add_argument("--device", default="cpu", help="cpu hoặc cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.amount < 1.0:
        raise ValueError("--amount phải nằm trong [0, 1).")
    if args.warmup < 0 or args.iterations < 1:
        raise ValueError("warmup >= 0 và iterations >= 1.")
    if args.threads is not None:
        if args.threads < 1:
            raise ValueError("--threads phải >= 1.")
        torch.set_num_threads(args.threads)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("Đã chọn CUDA nhưng torch.cuda.is_available() là False.")

    device = torch.device(args.device)
    # Đo tuần tự để RSS của baseline không bị cộng thêm bản sao pruned.
    baseline = build_model(args.num_classes)
    load_checkpoint(baseline, args.checkpoint)
    baseline.eval()
    baseline_parameters = parameter_report(baseline)
    size_before = save_model(baseline, Path(args.out).with_suffix(".fp32.pt"))
    baseline_benchmark = benchmark(
        baseline, device, args.input_size, args.warmup, args.iterations
    )
    del baseline
    gc.collect()

    pruned = build_model(args.num_classes)
    load_checkpoint(pruned, args.checkpoint)
    layer_info = apply_pruning(pruned, args.method, args.amount)
    pruned_parameters = parameter_report(pruned)
    size_after = save_model(pruned, Path(args.out))
    pruned_benchmark = benchmark(
        pruned, device, args.input_size, args.warmup, args.iterations
    )

    report = {
        "checkpoint": str(args.checkpoint),
        "method": args.method,
        "requested_amount": args.amount,
        "device": str(device),
        "torch_threads": torch.get_num_threads(),
        "input_size": [1, 3, args.input_size, args.input_size],
        "pruning": layer_info,
        "baseline": {
            "parameters": baseline_parameters,
            "size_mb": size_before,
            "benchmark": baseline_benchmark,
        },
        "pruned": {
            "parameters": pruned_parameters,
            "size_mb": size_after,
            "benchmark": pruned_benchmark,
        },
        "note": (
            "Mask/zero pruning không tự compact topology; file checkpoint và latency "
            "có thể gần như không đổi nếu runtime dùng dense kernel."
        ),
    }

    report_path = Path(args.report) if args.report else Path(args.out).with_suffix(".json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
