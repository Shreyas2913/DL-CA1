import torch
import torch.nn as nn
from torchvision import models

# Custom classification head on top of MobileNetV2
class SimpleXception(nn.Module):
    def __init__(self):
        super(SimpleXception, self).__init__()
        base = models.mobilenet_v2(pretrained=True)
        self.features = base.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1280, 2),  # Output for 2 classes
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x

# Initialize and export
model = SimpleXception()
model.eval()

dummy_input = torch.randn(1, 3, 299, 299)
torch.onnx.export(
    model,
    dummy_input,
    "xception_df.onnx",
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    opset_version=11
)

print("✅ ONNX model saved as xception_df.onnx")

