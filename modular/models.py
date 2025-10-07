import sys, os
sys.path.append(os.path.dirname(__file__))
from torchvision.models import ViT_B_16_Weights, vit_b_16
from vit_from_scratch import VisionTransformer
import torch

def create_vit_model(num_classes:int, seed:int=42):
    weights = ViT_B_16_Weights.DEFAULT
    transforms = weights.transforms()
    model = vit_b_16(weights = weights)

    for param in model.parameters():
        param.requires_grad = False

    torch.manual_seed(seed)
    model.heads = torch.nn.Sequential(
        torch.nn.Linear(
            in_features=768,
                out_features=num_classes
        )
    )

    return model, transforms

def create_vit_tiny_cpu(num_classes=5, img_size=224):
    """
    ViT Tiny - Siêu nhỏ cho CPU (1.2M params)
    Phù hợp: CIFAR-10, MNIST, datasets nhỏ
    Training time: ~5-10 phút/epoch trên CPU decent
    """
    return VisionTransformer(
        img_size=img_size,
        patch_size=64,
        in_channels=3,
        num_classes=num_classes,
        embed_dim=128,      # Rất nhỏ
        depth=4,            # Chỉ 4 layers
        num_heads=4,
        mlp_ratio=2,        # Giảm từ 4 xuống 2
        drop_rate=0.0,      # Tắt dropout khi model nhỏ
        attn_drop_rate=0.0
    )