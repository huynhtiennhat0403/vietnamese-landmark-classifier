from torchvision.models import ViT_B_16_Weights, vit_b_16
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