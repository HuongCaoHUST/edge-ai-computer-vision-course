# QAT native PyTorch

## 1. Phạm vi của baseline

QAT (Quantization-Aware Training) là bước fine-tune có mô phỏng lượng tử hóa. Nó không phải là calibration PTQ: calibration chỉ thu thập thống kê activation và không cập nhật weights, còn QAT dùng dữ liệu có nhãn để cập nhật weights.

QAT cho SSD MobileNet đầy đủ có thể yêu cầu chuẩn bị graph, fuse layer và hỗ trợ operator riêng của từng phiên bản PyTorch. Để bài thực hành dễ kiểm chứng, baseline này dùng **weight-only QAT**:

- weights Conv2d được fake-quantize trong forward;
- weights FP32 vẫn được optimizer cập nhật;
- activation và postprocess vẫn chạy FP32;
- sau fine-tune, weights được đóng gói thành INT8 như baseline PTQ.

Đây là baseline QAT về ảnh hưởng của sai số weights, chưa phải full integer execution.

## 2. Fake quantization

```text
w_fp32 → fake quantize → dequantize → convolution
```

Forward nhìn thấy sai số gần giống INT8, còn optimizer vẫn nhận gradient thông qua phép xấp xỉ Straight-Through Estimator. Nhờ đó model có thể học cách giảm độ nhạy với sai số lượng tử hóa.

QAT vẫn cần dữ liệu chạy qua model để observer ước lượng range trong giai đoạn đầu, nhưng đây là **observer initialization trong quá trình training**, không phải calibration PTQ độc lập. QAT sử dụng dữ liệu có nhãn để fine-tune weights; PTQ calibration chỉ thu thập thống kê và không cập nhật weights.

## 3. Module Conv2d có fake quantizer

```python
import torch
from torch import nn
from torch.ao.quantization import FakeQuantize, MovingAverageMinMaxObserver
from torch.nn import functional as F


class FakeQuantConv2d(nn.Conv2d):
    def __init__(self, source: nn.Conv2d):
        super().__init__(
            source.in_channels, source.out_channels,
            source.kernel_size, source.stride, source.padding,
            source.dilation, source.groups,
            source.bias is not None, source.padding_mode,
        )
        self.load_state_dict(source.state_dict())
        self.weight_fake_quant = FakeQuantize(
            observer=MovingAverageMinMaxObserver,
            quant_min=-127,
            quant_max=127,
            dtype=torch.qint8,
            qscheme=torch.per_tensor_symmetric,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.weight_fake_quant(self.weight)
        return F.conv2d(
            x, weight, self.bias, self.stride,
            self.padding, self.dilation, self.groups,
        )
```

## 4. Thay thế các convolution

```python
import copy

def replace_with_fake_quant(module: nn.Module) -> None:
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Conv2d):
            setattr(module, name, FakeQuantConv2d(child))
        else:
            replace_with_fake_quant(child)


model_qat = copy.deepcopy(model_fp32).to(device)
replace_with_fake_quant(model_qat)
```

Chỉ thay thế `Conv2d` ở baseline này. BatchNorm, activation, detection head logic và NMS được giữ nguyên để giảm số biến cần kiểm soát.

## 5. Fine-tune bằng vòng lặp PyTorch

```python
optimizer = torch.optim.SGD(
    model_qat.parameters(),
    lr=1e-5,
    momentum=0.9,
    weight_decay=5e-4,
)

for epoch in range(3):
    model_qat.train()
    for images, targets in train_loader:
        images = [image.to(device) for image in images]
        targets = [
            {key: value.to(device) if hasattr(value, "to") else value
             for key, value in target.items()}
            for target in targets
        ]

        optimizer.zero_grad(set_to_none=True)
        loss_dict = model_qat(images, targets)
        loss = sum(loss_dict.values())
        loss.backward()
        optimizer.step()

    model_qat.eval()
    metrics = evaluate_detection(model_qat, val_loader, device)
    print(epoch, metrics)
```

QAT nên bắt đầu từ checkpoint FP32 đã hội tụ, dùng learning rate nhỏ và chọn checkpoint theo mAP validation. Không chọn chỉ dựa vào training loss.

## 6. Freeze observer và xuất weights INT8

Sau một hoặc hai epoch đầu, có thể dừng việc cập nhật range của fake quantizer:

```python
for module in model_qat.modules():
    if isinstance(module, FakeQuantize):
        module.disable_observer()
```

Sau khi fine-tune, dùng module `WeightOnlyInt8Conv2d` trong trang PTQ để đóng gói weights. Model sau bước này có thể đánh giá chất lượng và kích thước giống baseline PTQ.

## 7. Giới hạn cần ghi trong báo cáo

Baseline QAT này không chứng minh:

- activation đã chạy INT8;
- toàn bộ SSD MobileNet chạy integer;
- latency sẽ giảm trên CPU/NPU;
- mọi operator của model đều được backend INT8 hỗ trợ.

Nó chỉ kiểm chứng liệu fine-tune với fake-quantized weights có giúp giảm `ΔmAP` hay không. Đây là baseline phù hợp trước khi mở rộng sang full static INT8.
