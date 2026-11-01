"""Auditable static PTQ baseline implemented with PyTorch only.

The baseline follows:

    trained FP32 model -> calibration -> INT8 fake-quantized inference

It is deliberately simple. It quantizes Conv2d weights and activations per
tensor, then executes the convolution with dequantized tensors in PyTorch.
It is useful for checking accuracy and the calibration flow; it does not claim
hardware integer-kernel acceleration.
"""

from __future__ import annotations

import copy
from typing import Iterable

import torch
from torch import nn
from torch.ao.quantization import FakeQuantize, MovingAverageMinMaxObserver
from torch.nn import functional as F


class MinMaxObserver:
    """Small CPU-side observer for reproducible per-tensor calibration."""

    def __init__(self) -> None:
        self.minimum: float | None = None
        self.maximum: float | None = None

    @torch.no_grad()
    def observe(self, value: torch.Tensor) -> None:
        value_min = float(value.detach().amin().cpu())
        value_max = float(value.detach().amax().cpu())
        self.minimum = value_min if self.minimum is None else min(self.minimum, value_min)
        self.maximum = value_max if self.maximum is None else max(self.maximum, value_max)

    def affine_qparams(self, qmin: int = 0, qmax: int = 255) -> tuple[float, int]:
        if self.minimum is None or self.maximum is None:
            raise RuntimeError("Observer has no data; run calibration first")
        minimum = min(self.minimum, 0.0)
        maximum = max(self.maximum, 0.0)
        scale = (maximum - minimum) / float(qmax - qmin)
        scale = max(scale, 1e-8)
        zero_point = round(qmin - minimum / scale)
        zero_point = max(qmin, min(qmax, zero_point))
        return scale, zero_point


def fake_quantize(
    value: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
    qmin: int = 0,
    qmax: int = 255,
) -> torch.Tensor:
    quantized = torch.clamp(
        torch.round(value / scale) + zero_point,
        qmin,
        qmax,
    )
    return (quantized - zero_point) * scale


class WeightOnlyInt8Conv2d(nn.Module):
    """Conv2d with per-tensor symmetric INT8 weights and FP32 execution."""

    def __init__(self, source: nn.Conv2d) -> None:
        super().__init__()
        weight = source.weight.detach().cpu()
        max_abs = weight.abs().max()
        scale = max_abs / 127 if max_abs > 0 else torch.tensor(1.0)
        qweight = torch.clamp(torch.round(weight / scale), -127, 127).to(torch.int8)

        self.register_buffer("qweight", qweight)
        self.register_buffer("scale", scale.reshape(()).float())
        if source.bias is not None:
            self.register_buffer("bias", source.bias.detach().clone())
        else:
            self.bias = None

        self.stride = source.stride
        self.padding = source.padding
        self.dilation = source.dilation
        self.groups = source.groups

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        weight = self.qweight.float() * self.scale
        return F.conv2d(
            value,
            weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class StaticPTQConv2d(nn.Module):
    """Conv2d with calibrated per-tensor INT8 weights and activations."""

    def __init__(self, source: nn.Conv2d) -> None:
        super().__init__()
        self.register_buffer("weight_fp32", source.weight.detach().clone())
        if source.bias is not None:
            self.register_buffer("bias", source.bias.detach().clone())
        else:
            self.bias = None

        self.stride = source.stride
        self.padding = source.padding
        self.dilation = source.dilation
        self.groups = source.groups
        self.input_observer = MinMaxObserver()
        self.output_observer = MinMaxObserver()
        self.calibrated = False

    def _forward_fp32(self, value: torch.Tensor) -> torch.Tensor:
        return F.conv2d(
            value,
            self.weight_fp32,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if not self.calibrated:
            self.input_observer.observe(value)
            output = self._forward_fp32(value)
            self.output_observer.observe(output)
            return output

        input_value = fake_quantize(value, self.input_scale, self.input_zero_point)
        weight = self.qweight.float() * self.weight_scale
        output = F.conv2d(
            input_value,
            weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
        return fake_quantize(output, self.output_scale, self.output_zero_point)

    @torch.no_grad()
    def finalize(self) -> None:
        input_scale, input_zero_point = self.input_observer.affine_qparams()
        output_scale, output_zero_point = self.output_observer.affine_qparams()

        max_abs = float(self.weight_fp32.abs().max().cpu())
        weight_scale = max(max_abs / 127.0, 1e-8)
        qweight = torch.clamp(
            torch.round(self.weight_fp32 / weight_scale), -127, 127
        ).to(torch.int8)

        self.register_buffer("qweight", qweight)
        self.register_buffer("weight_scale", torch.tensor(weight_scale))
        self.register_buffer("input_scale", torch.tensor(input_scale))
        self.register_buffer("input_zero_point", torch.tensor(float(input_zero_point)))
        self.register_buffer("output_scale", torch.tensor(output_scale))
        self.register_buffer("output_zero_point", torch.tensor(float(output_zero_point)))
        del self._buffers["weight_fp32"]
        self.calibrated = True


class FakeQuantConv2d(nn.Conv2d):
    """Conv2d whose weights are fake-quantized during QAT."""

    def __init__(self, source: nn.Conv2d) -> None:
        super().__init__(
            source.in_channels,
            source.out_channels,
            source.kernel_size,
            source.stride,
            source.padding,
            source.dilation,
            source.groups,
            source.bias is not None,
            source.padding_mode,
        )
        self.load_state_dict(source.state_dict())
        self.weight_fake_quant = FakeQuantize(
            observer=MovingAverageMinMaxObserver,
            quant_min=-127,
            quant_max=127,
            dtype=torch.qint8,
            qscheme=torch.per_tensor_symmetric,
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        weight = self.weight_fake_quant(self.weight)
        return F.conv2d(
            value,
            weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


def _replace_conv2d(module: nn.Module) -> int:
    replaced = 0
    for name, child in list(module.named_children()):
        if type(child) is nn.Conv2d:
            setattr(module, name, StaticPTQConv2d(child))
            replaced += 1
        else:
            replaced += _replace_conv2d(child)
    return replaced


def prepare_static_ptq(model: nn.Module) -> tuple[nn.Module, int]:
    """Return a CPU copy prepared to collect calibration statistics."""

    result = copy.deepcopy(model).cpu().eval()
    return result, _replace_conv2d(result)


@torch.no_grad()
def calibrate(
    model: nn.Module,
    calibration_loader: Iterable,
    device: torch.device,
    limit: int = 300,
) -> int:
    """Run representative data through the prepared model."""

    model.eval()
    seen = 0
    for images, _targets in calibration_loader:
        images = [image.to(device) for image in images]
        model(images)
        seen += len(images)
        if seen >= limit:
            break
    return seen


@torch.no_grad()
def convert_static_ptq(model: nn.Module) -> None:
    """Freeze all collected ranges and pack weights as INT8."""

    for module in model.modules():
        if isinstance(module, StaticPTQConv2d):
            module.finalize()


def prepare_qat(model: nn.Module) -> tuple[nn.Module, int]:
    """Return a trainable copy with weight fake quantizers."""

    result = copy.deepcopy(model)
    replaced = 0
    for name, child in list(result.named_children()):
        if type(child) is nn.Conv2d:
            setattr(result, name, FakeQuantConv2d(child))
            replaced += 1
        else:
            nested, nested_count = prepare_qat(child)
            setattr(result, name, nested)
            replaced += nested_count
    return result, replaced


def freeze_qat_observers(model: nn.Module) -> None:
    """Freeze weight ranges after the first QAT epochs."""

    for module in model.modules():
        if isinstance(module, FakeQuantize):
            module.disable_observer()


def pack_qat_weights(model: nn.Module) -> int:
    """Replace QAT Conv2d modules with weight-only INT8 modules."""

    packed = 0
    for name, child in list(model.named_children()):
        if isinstance(child, FakeQuantConv2d):
            setattr(model, name, WeightOnlyInt8Conv2d(child))
            packed += 1
        else:
            packed += pack_qat_weights(child)
    return packed


def count_int8_weights(model: nn.Module) -> int:
    """Return the number of packed INT8 weight values."""

    return sum(
        int(module.qweight.numel())
        for module in model.modules()
        if isinstance(module, StaticPTQConv2d) and hasattr(module, "qweight")
    )
