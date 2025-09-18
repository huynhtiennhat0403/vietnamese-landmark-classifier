import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

NUM_WORKERS = 4
BATCH_SIZE = 32

def create_dataloaders(train_dir:str, test_dir:str, transform: transforms,batch_size:int = BATCH_SIZE,  num_workers: int = NUM_WORKERS):
    train_data = datasets.ImageFolder(train_dir, transform=transform)
    test_data = datasets.ImageFolder(test_dir, transform=transform)

    class_names = train_data.classes
    target = test_data.targets
    train_dataloader = DataLoader(
        train_data, 
        batch_size= batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    test_dataloader = DataLoader(
        test_data, 
        batch_size, 
        False,
        num_workers=num_workers
    )

    return train_dataloader, test_dataloader, class_names, target