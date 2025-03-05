import os
import json
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
import random
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
# Grad-CAM utilities from pytorch-grad-cam
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import numpy as np
from model import CNN

plt.style.use('seaborn-v0_8-darkgrid')

# ---------------------------
# Hyperparameters and paths
# ---------------------------
NUM_EPOCHS = 10
BATCH_SIZE = 1028
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_LOADER_SEED = 42  # Fixed seed for reproducibility during retraining
SAVED_TENSORSETS_DIR = "./saved_tensorsets"
RESULTS_DIR = "./kmnist/results_bayes_opt"
BEST_RESULTS_FILE = os.path.join(RESULTS_DIR, "problem1_bayes_opt_results_CNN_v1.json")

# ---------------------------
# Load best hyperparameters from results file
# ---------------------------
with open(BEST_RESULTS_FILE, "r") as f:
    best_results = json.load(f)
best_params = best_results["best_params"]



# ---------------------------
# Data loading functions
# ---------------------------
def load_tensor_datasets(save_dir=SAVED_TENSORSETS_DIR):
    train_tds = torch.load("./kmnist/results_augmentation_search/kmnist_train_augmented_tensorset.pth")
    val_tds = torch.load("./kmnist/saved_tensorsets/kmnist_val_tensorset.pth")
    test_tds = torch.load("./kmnist/saved_tensorsets/kmnist_test_tensorset.pth")
    return train_tds, val_tds, test_tds

def get_dataloaders(train_tds, val_tds, test_tds, batch_size=BATCH_SIZE, shuffle_seed=DATA_LOADER_SEED):
    g = torch.Generator()
    g.manual_seed(shuffle_seed)
    train_loader = DataLoader(train_tds, batch_size=batch_size, shuffle=True, generator=g)
    val_loader = DataLoader(val_tds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_tds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader

def generate_lr_schedule(lr_base, lr_peak, lr_end, num_warmup_epochs, total_epochs):
    schedule = []
    for epoch in range(total_epochs):
        if epoch < num_warmup_epochs:
            lr = lr_base + (lr_peak - lr_base) * (epoch+1) / num_warmup_epochs
        else:
            cosine_decay = 0.5 * (1 + math.cos(math.pi * (epoch - num_warmup_epochs) / (total_epochs - num_warmup_epochs)))
            lr = lr_end + (lr_peak - lr_end) * cosine_decay
        schedule.append(lr)
    return schedule

# ---------------------------
# Training and Evaluation Function
# ---------------------------
def train_and_evaluate(seed, best_params):
    torch.manual_seed(seed)
    random.seed(seed)
    train_tds, val_tds, test_tds = load_tensor_datasets()
    train_loader, val_loader, test_loader = get_dataloaders(train_tds, val_tds, test_tds, batch_size=BATCH_SIZE, shuffle_seed=seed)
    
    lr_peak = best_params["lr_peak"]
    lr_base = best_params["base_frac"] * lr_peak
    lr_end = best_params["end_frac"] * lr_peak
    num_warmup_epochs = max(1, int(round(best_params["num_warmup_epochs"])))
    lr_schedule = generate_lr_schedule(lr_base, lr_peak, lr_end, num_warmup_epochs, NUM_EPOCHS)
    wd_schedule = [best_params["weight_decay"]] * NUM_EPOCHS

    model = CNN(num_classes=10).to(DEVICE)
    # Reset model parameters using default initialization
    model.apply(lambda m: m.reset_parameters() if hasattr(m, 'reset_parameters') else None)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr_schedule[0], weight_decay=best_params["weight_decay"],
                            betas=(best_params["b1"], best_params["b2"]))
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(1, NUM_EPOCHS+1):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
    model.eval()
    tot_correct, tot_samples = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            tot_correct += (preds == labels).sum().item()
            tot_samples += images.size(0)
    acc = tot_correct / tot_samples
    return acc, model, test_loader

# ---------------------------
# Helper for Grad-CAM Visualization: Convert grayscale to 3-channel image
# ---------------------------
def process_img(img):
    img_np = img.squeeze().numpy()
    if img_np.ndim == 2:
        img_np = np.stack([img_np]*3, axis=-1)
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
    return img_np

# ---------------------------
# Interactive Visualization: Grad-CAM
# ---------------------------
def interactive_visualize(model, test_loader, device=DEVICE, num_samples_per_class=2):
    target_layer = model.conv2  # Use conv2 as target layer
    cam = GradCAM(model=model, target_layers=[target_layer])
    
    def collect_samples():
        misclassified = {}
        correct = {}
        model.eval()
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                preds = outputs.argmax(dim=1)
                for img, true, pred in zip(images, labels, preds):
                    true = true.item()
                    pred = pred.item()
                    if true == pred:
                        if true not in correct or len(correct[true]) < num_samples_per_class:
                            correct.setdefault(true, []).append(img.cpu())
                    else:
                        if true not in misclassified or len(misclassified[true]) < num_samples_per_class:
                            misclassified.setdefault(true, []).append((img.cpu(), pred))
                # Once we've gathered enough, break out
                if len(misclassified) >= 10 and len(correct) >= 10:
                    break
        return misclassified, correct

    def visualize_samples(samples, title_prefix):
        # For each class, plot all samples (each in a subplot)
        fig, axes = plt.subplots(1, len(samples), figsize=(4*len(samples), 4))
        if len(samples) == 1:
            axes = [axes]
        for ax, cls in zip(axes, sorted(samples.keys())):
            # Take the first sample for this class
            sample = samples[cls][0]
            if isinstance(sample, tuple):
                img, pred = sample
            else:
                img = sample
                pred = None
            img_np = process_img(img)
            input_tensor = img.unsqueeze(0).to(device)
            grayscale_cam = cam(input_tensor=input_tensor)[0, :]
            cam_image = show_cam_on_image(img_np, grayscale_cam, use_rgb=False)
            ax.imshow(cam_image)
            if pred is not None:
                ax.set_title(f"{title_prefix}\nTrue={cls}, Pred={pred}")
            else:
                ax.set_title(f"{title_prefix}\nClass {cls}")
            ax.axis("off")
        plt.tight_layout()
        plt.show()

    while True:
        misclassified, correct = collect_samples()
        v = input("Press 1 to view misclassified samples, 2 to view correctly classified samples, 3 to exit visualization: ").strip()
        if v == "1":
            visualize_samples(misclassified, "Misclassified")
        elif v == "2":
            visualize_samples(correct, "Correctly Classified")
        elif v == "3":
            break

# ---------------------------
# Main: Retrain and Visualize
# ---------------------------
if __name__ == "__main__":
    # Choose whether to run a single training run for visualization only.
    VISUALIZE_ONLY = False

    if VISUALIZE_ONLY:
        seed = 42  # Fixed seed
        acc, model, test_loader = train_and_evaluate(seed, best_params)
        print(f"Test Accuracy = {acc:.4f}")
        interactive_visualize(model, test_loader, device=DEVICE, num_samples_per_class=1)
    else:
        num_runs = 5
        accuracies = []
        models = []
        for run in range(num_runs):
            seed = 42 + run
            acc, model, test_loader = train_and_evaluate(seed, best_params)
            print(f"Run {run+1}: Test Accuracy = {acc:.4f}")
            accuracies.append(acc)
            models.append(model)
        avg_acc = sum(accuracies)/len(accuracies)
        var_acc = sum((x - avg_acc) ** 2 for x in accuracies)/len(accuracies)
        print(f"\nAverage Test Accuracy over {num_runs} runs: {avg_acc:.4f}")
        print(f"Variance over {num_runs} runs: {var_acc:.6f}")
        
        best_run = max(range(num_runs), key=lambda i: accuracies[i])
        print(f"Visualizing results for Run {best_run+1} (Accuracy = {accuracies[best_run]:.4f})")
        interactive_visualize(models[best_run], test_loader, device=DEVICE, num_samples_per_class=1)