import matplotlib.pyplot as plt
from typing import Dict, List
from tqdm.auto import tqdm
import torch
import torchvision
from torchmetrics import ConfusionMatrix
from pathlib import Path
from PIL import Image
from timeit import default_timer as timer
from mlxtend.plotting import plot_confusion_matrix
import pandas as pd
import os

def plot_results(results: Dict[str, List], 
                 epochs: int, 
                 save_path: str,
                 show_plot: bool = True):
    """
    Vẽ biểu đồ training/testing loss và accuracy
    
    Args:
        results: Dictionary chứa 'train_loss', 'test_loss', 'train_acc', 'test_acc'
        epochs: Số lượng epochs
        save_path: Đường dẫn để lưu ảnh (nếu None thì không lưu)
                   Ví dụ: 'results/training_plot.png' hoặc 'training_plot'
        show_plot: Có hiển thị plot hay không (default: True)
    
    Returns:
        fig: Matplotlib figure object
    """
    fig = plt.figure(figsize=(12, 6))

    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs + 1), results['train_loss'], label='Training loss', marker='o')
    plt.plot(range(1, epochs + 1), results['test_loss'], label='Testing loss', marker='s')
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training and Testing Loss', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs + 1), results['train_acc'], label='Training accuracy', marker='o')
    plt.plot(range(1, epochs + 1), results['test_acc'], label='Testing accuracy', marker='s')
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title('Training and Testing Accuracy', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    # Lưu ảnh nếu save_path được cung cấp
    if save_path is not None:
        # Tự động thêm extension nếu chưa có
        if not save_path.endswith(('.png', '.jpg', '.jpeg', '.pdf', '.svg')):
            save_path = f"{save_path}.png"
        
        # Tạo thư mục nếu chưa tồn tại
        save_dir = os.path.dirname(save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # Lưu với quality cao
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✅ Plot saved to: {save_path}")
    
    # Hiển thị plot
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    return fig

def predict_and_plot_confusion_matrix(dataloader: torch.utils.data.DataLoader, 
                                      model: torch.nn.Module, 
                                      class_names: List[str],
                                      targets,
                                      save_path: str,
                                      device: str = 'cpu',
                                      show_plot: bool = True):
    """
    Dự đoán và vẽ confusion matrix
    
    Args:
        dataloader: DataLoader chứa dữ liệu test
        model: Model để dự đoán
        class_names: List tên các classes
        targets: Ground truth labels
        device: Device để chạy model ('cpu' hoặc 'cuda')
        save_path: Đường dẫn để lưu ảnh (nếu None thì không lưu)
                   Ví dụ: 'results/confusion_matrix.png'
        show_plot: Có hiển thị plot hay không (default: True)
    
    Returns:
        fig: Matplotlib figure object
        ax: Matplotlib axes object
        y_preds_tensor: Tensor chứa predictions
    """
    targets = torch.as_tensor(targets)
    y_preds = []
    
    model.eval()
    with torch.inference_mode():
        for X, y in tqdm(dataloader, desc="Making predictions"):
            X, y = X.to(device), y.to(device)
            y_logit = model(X)
            y_pred = torch.softmax(y_logit, 1).argmax(1)
            y_preds.append(y_pred.cpu())

    y_preds_tensor = torch.cat(y_preds)

    # Tạo confusion matrix
    confmat = ConfusionMatrix(task='multiclass', num_classes=len(class_names))
    confmat_tensor = confmat(preds=y_preds_tensor, target=targets)

    # Plot confusion matrix
    fig, ax = plot_confusion_matrix(
        conf_mat=confmat_tensor.numpy(),
        class_names=class_names,
        figsize=(10, 7)
    )
    
    # Thêm title
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()

    # Lưu ảnh nếu save_path được cung cấp
    if save_path is not None:
        # Tự động thêm extension nếu chưa có
        if not save_path.endswith(('.png', '.jpg', '.jpeg', '.pdf', '.svg')):
            save_path = f"{save_path}.png"
        
        # Tạo thư mục nếu chưa tồn tại
        save_dir = os.path.dirname(save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # Lưu với quality cao
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✅ Confusion matrix saved to: {save_path}")
    
    # Hiển thị plot
    if show_plot:
        plt.show()
    else:
        plt.close()

    return fig, ax, y_preds_tensor

def predict_and_store(test_data_paths: List[Path], model: torch.nn.Module, transforms: torchvision.transforms,class_names: List[str], device:str = 'cpu'):
    preds = []

    for path in tqdm(test_data_paths):
        pred = {}
        start_time = timer()
        pred['path'] = path
        
        class_name = path.parent.stem
        pred['class'] = class_name

        image = Image.open(path).convert("RGB")
        transformed_image = transforms(image).unsqueeze(0).to(device)

        model.to(device)
        model.eval()
        with torch.inference_mode():
            y_logit = model(transformed_image)
            y_prob = torch.softmax(y_logit, 1)
            pred['prob'] = round(y_prob.unsqueeze(0).max().item(),4)
            y_label = y_prob.argmax(1)
            predicted_class = class_names[y_label.cpu()]

            end_time = timer()
            pred['time_for_pred'] = round(end_time - start_time, 4)
        
        pred['correct'] = predicted_class == class_name

        preds.append(pred)

    return pd.DataFrame(preds)

    

        