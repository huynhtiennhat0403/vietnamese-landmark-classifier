import os
import torch
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from modular import data_setup, models, engine, utils

if __name__ == '__main__':
    # Setup hyperparameters
    NUM_EPOCHS = 3
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001

    # Đường dẫn tuyệt đối tới thư mục chứa train.py
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    # Tạo đường dẫn tuyệt đối tới data/train và data/test
    train_dir = os.path.join(BASE_DIR, "..", "data", "train")
    test_dir = os.path.join(BASE_DIR, "..", "data", "test")

    # Num classes
    num_classes = len([f for f in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, f))])

    # Setup target device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create model with help from model_builder.py
    model, transform = models.create_vit_model(
        num_classes=num_classes,
        seed=42
    )

    # Create DataLoaders with help from data_setup.py
    train_dataloader, test_dataloader, class_names, targets = data_setup.create_dataloaders(
        train_dir,
        test_dir,
        transform,
        BATCH_SIZE
    )

    # Set loss and optimizer
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                lr=LEARNING_RATE)

    # Start training with help from engine.py
    results = engine.train(model, train_dataloader, test_dataloader, optimizer, loss_fn, epochs=NUM_EPOCHS, device='cpu')

    # Save the model with help from utils.py
    utils.save_model(model=model,
                    target_dir="models",
                    model_name="vietnamese_landmark_vit.pth")