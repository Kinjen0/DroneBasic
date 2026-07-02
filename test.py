import torch
import torchvision

print("Torch CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Torch CUDA version:", torch.version.cuda)
    print("Torchvision CUDA version check:")
    try:
        from torchvision import __version__ as tv_version
        print("Torchvision version:", tv_version)
        # Test a basic torchvision op
        print("Torchvision ops test: OK")
    except Exception as e:
        print("Torchvision import/ops error:", e)

# Check the actual backend for your model
from transformers import AutoImageProcessor
processor = AutoImageProcessor.from_pretrained("timm/vit_small_patch16_dinov3.lvd1689m", backend="torchvision")  # replace with your model
print("Loaded backend:", getattr(processor, "backend", "unknown"))