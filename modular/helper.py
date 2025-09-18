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

def plot_results(results: Dict[str, List], epochs: int):
    fig = plt.figure(figsize=(12,6))

    plt.subplot(1,2,1)
    plt.plot(range(1, epochs + 1), results['train_loss'], label = 'Training loss')
    plt.plot(range(1, epochs + 1), results['test_loss'], label = 'Testing loss')
    plt.xlabel('Epochs')
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(range(1, epochs + 1), results['train_acc'], label = 'Training accuracy')
    plt.plot(range(1, epochs + 1), results['test_acc'], label = 'Testing accuracy')
    plt.xlabel('Epochs')
    plt.legend()

def predict_and_plot_confusion_matrix(dataloader: torch.utils.data.DataLoader, 
                                      model: torch.nn.Module, 
                                      class_names: List[str],
                                      targets,
                                      device: str = 'cpu'):
    targets = torch.as_tensor(targets)
    y_preds = []
    model.eval()
    with torch.inference_mode():
        for X, y in tqdm(dataloader, desc="Making predictions"):
            X, y = X.to(device), y.to(device)
            y_logit = model(X)
            y_pred = torch.softmax(y_logit,1).argmax(1)
            y_preds.append(y_pred.cpu())

    y_preds_tensor = torch.cat(y_preds)

    confmat = ConfusionMatrix(task='multiclass', num_classes=len(class_names))
    confmat_tensor = confmat(preds = y_preds_tensor, target = targets)

    fig, ax = plot_confusion_matrix(
        conf_mat=confmat_tensor.numpy(),
        class_names=class_names,
        figsize=(10,7)
    )

    return fig, ax

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

    

        