import os
import copy
import json
import random
import time
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from bayes_opt import BayesianOptimization  # pip install bayesian-optimization

plt.style.use('seaborn-v0_8-darkgrid')

# =============== Hyperparameters & Global Settings ===============
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVED_TENSORSETS_DIR = "./kmnist/saved_tensorsets"
RESULTS_DIR = "./kmnist/results_bayes_opt"
os.makedirs(RESULTS_DIR, exist_ok=True)

DATA_LOADER_SEED = 12345  # fixed seed for reproducibility
THRESH_ZERO = 1e-12

BEST_PREFIX_CACHE = {}
GLOBAL_EVAL_COUNT = 0

tqdm_kwargs = dict(
    leave=False,
    dynamic_ncols=True,
    bar_format="{l_bar}{bar} {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
)

# ---------------------------
# Data Loading Functions
# ---------------------------
def load_tensor_datasets(save_dir=SAVED_TENSORSETS_DIR):
    tqdm.write(f"[load_tensor_datasets] Loading KMNIST TensorDatasets from '{save_dir}'...")
    train_tds = torch.load("./kmnist/results_augmentation_search/kmnist_train_augmented_tensorset.pth")
    val_tds = torch.load(os.path.join("./kmnist/saved_tensorsets", "kmnist_val_tensorset.pth"))
    test_tds = torch.load(os.path.join("./kmnist/saved_tensorsets", "kmnist_test_tensorset.pth"))
    tqdm.write(f"[load_tensor_datasets] Done. (Train={len(train_tds)}, Val={len(val_tds)}, Test={len(test_tds)})")
    return train_tds, val_tds, test_tds

def get_dataloaders(train_ds, val_ds, test_ds, batch_size=32, shuffle_seed=DATA_LOADER_SEED):
    tqdm.write(f"[get_dataloaders] Creating DataLoaders (batch_size={batch_size}, seed={shuffle_seed})...")
    g = torch.Generator()
    g.manual_seed(shuffle_seed)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=g)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader

def compute_loss(model, loader, criterion, device=DEVICE):
    model.eval()
    total_loss, total_samples = 0.0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            total_samples += images.size(0)
    return total_loss / total_samples if total_samples > 0 else float('inf')

# ---------------------------
# CNN Architecture
# ---------------------------
class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(64)
        self.pool  = nn.MaxPool2d(2, 2)
        self.fc1   = nn.Linear(64 * 7 * 7, 1024)
        self.bn3   = nn.BatchNorm1d(1024)
        self.dropout = nn.Dropout(0.2)
        self.fc2   = nn.Linear(1024, 512)
        self.bn4   = nn.BatchNorm1d(512)
        self.fc3   = nn.Linear(512, 256)
        self.bn5   = nn.BatchNorm1d(256)
        self.fc4   = nn.Linear(256, 128)
        self.bn6   = nn.BatchNorm1d(128)
        self.fc5   = nn.Linear(128, num_classes)
    
    def forward(self, x):
        x = F.gelu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = F.gelu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = F.gelu(self.bn3(self.fc1(x)))
        x = self.dropout(x)
        x = F.gelu(self.bn4(self.fc2(x)))
        x = self.dropout(x)
        x = F.gelu(self.bn5(self.fc3(x)))
        x = self.dropout(x)
        x = F.gelu(self.bn6(self.fc4(x)))
        x = self.dropout(x)
        x = self.fc5(x)
        return x

# ---------------------------
# Fixed LR Schedule: Linear Warmup + Cosine Annealing
# ---------------------------
def generate_lr_schedule(lr_base, lr_peak, lr_end, num_warmup_epochs, total_epochs):
    schedule = []
    for epoch in range(total_epochs):
        if epoch < num_warmup_epochs:
            lr = lr_base + (lr_peak - lr_base) * (epoch + 1) / num_warmup_epochs
        else:
            cosine_decay = 0.5 * (1 + math.cos(math.pi * (epoch - num_warmup_epochs) / (total_epochs - num_warmup_epochs)))
            lr = lr_end + (lr_peak - lr_end) * cosine_decay
        schedule.append(lr)
    return schedule

# ---------------------------
# Batch Size Scheduler Function
# ---------------------------
def batch_size_schedule(epoch, total_epochs, bs_initial, bs_final, bs_decay_start, bs_decay_rate):
    """
    Returns the current batch size for a given epoch.
    - Before bs_decay_start: returns bs_initial.
    - After bs_decay_start: uses a linear decay modified by bs_decay_rate.
    """
    if epoch < bs_decay_start:
        return bs_initial
    else:
        progress = (epoch - bs_decay_start) / (total_epochs - bs_decay_start)
        effective_progress = min(1.0, progress * bs_decay_rate)
        current_bs = bs_initial - effective_progress * (bs_initial - bs_final)
        return int(max(current_bs, bs_final))

# ---------------------------
# Evaluation Function (Full Training) with Adaptive Batch Size & Weight Decay
# ---------------------------
def evaluate_schedule_full(lr_schedule, constant_wd, init_state_dict, train_tds, val_tds, test_tds,
                           total_epochs, device, bs_initial, bs_final, bs_decay_start, bs_decay_rate, wd_bs_scaling):
    # Create fixed validation and test loaders (using a small batch size for evaluation)
    val_loader = DataLoader(val_tds, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_tds, batch_size=32, shuffle=False)
    
    num_classes = int(train_tds.tensors[1].max().item() + 1)
    model = CNN(num_classes=num_classes).to(device)
    model.load_state_dict(init_state_dict)
    criterion = nn.CrossEntropyLoss()
    
    optimizer = optim.AdamW(model.parameters(), lr=lr_schedule[0],
                            weight_decay=constant_wd, betas=(0.85, 0.999))
    scaler = torch.cuda.amp.GradScaler()

    last_bs = None
    train_loader = None
    
    # Training loop over total_epochs with dynamic batch size and weight decay
    for e in range(1, total_epochs + 1):
        current_lr = lr_schedule[e - 1]
        current_bs = batch_size_schedule(e, total_epochs, bs_initial, bs_final, bs_decay_start, bs_decay_rate)
        wd_current = constant_wd * (1 + wd_bs_scaling * (1 - (current_bs / bs_initial)))
        # Only recreate DataLoader if batch size has changed
        if last_bs is None or current_bs != last_bs:
            train_loader, _, _ = get_dataloaders(train_tds, val_tds, test_tds, batch_size=current_bs)
            last_bs = current_bs

        for pg in optimizer.param_groups:
            pg['lr'] = current_lr
            pg['weight_decay'] = wd_current
        
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
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
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            tot_correct += (preds == labels).sum().item()
            tot_samples += images.size(0)
    acc = tot_correct / tot_samples
    return acc, copy.deepcopy(model.state_dict())

# ---------------------------
# Saving Experiment Results
# ---------------------------
def save_experiment_results(method_name, meta, best_state, init_state_dict):
    init_path = os.path.join(RESULTS_DIR, f"problem1_{method_name}_init_state_dict.pth")
    torch.save(init_state_dict, init_path)
    best_path = os.path.join(RESULTS_DIR, f"problem1_{method_name}_best_state_dict.pth")
    if best_state is not None:
        torch.save(best_state, best_path)
    meta["init_state_file"] = init_path
    meta["best_state_file"] = best_path
    json_path = os.path.join(RESULTS_DIR, f"problem1_{method_name}_results.json")
    with open(json_path, "w") as f:
        json.dump(meta, f, indent=4)
    tqdm.write(f"[{method_name.upper()}] Results saved to {json_path}.")

# ---------------------------
# Main Pipeline with Bayesian Optimization (Including Batch Scheduler)
# ---------------------------
def main():
    tqdm.write(f"[main] Setting up results folder: '{RESULTS_DIR}'")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    global GLOBAL_EVAL_COUNT
    GLOBAL_EVAL_COUNT = 0

    tqdm.write("[main] Loading dataset + building initial CNN model.")
    train_tds, val_tds, test_tds = load_tensor_datasets(SAVED_TENSORSETS_DIR)
    num_classes = int(train_tds.tensors[1].max().item() + 1)
    torch.manual_seed(42)
    init_model = CNN(num_classes=num_classes)
    init_sd = copy.deepcopy(init_model.state_dict())

    # Bayesian optimization will tune:
    # - lr_base, lr_peak, num_warmup_epochs (for LR schedule)
    # - total_epochs (total training epochs)
    # - bs_decay_start (epoch at which batch size reduction starts)
    # - bs_decay_rate (controls how fast batch size decays)
    # - wd_bs_scaling (scales weight decay with respect to batch size reduction)
    def objective(lr_base, lr_peak, num_warmup_epochs, total_epochs, bs_decay_start, bs_decay_rate, wd_bs_scaling,bs_final):
        total_epochs = int(round(total_epochs))
        num_warmup_epochs = max(1, int(round(num_warmup_epochs)))
        bs_decay_start = max(1, int(round(bs_decay_start)))
        bs_final = int(round(bs_final))
        lr_end = 1e-6
        lr_schedule = generate_lr_schedule(lr_base, lr_peak, lr_end, num_warmup_epochs, total_epochs)
        constant_wd = 1e-5
        acc, _ = evaluate_schedule_full(lr_schedule, constant_wd, init_sd,
                                        train_tds, val_tds, test_tds,
                                        total_epochs=total_epochs,
                                        device=DEVICE,
                                        bs_initial=1028, bs_final=bs_final,
                                        bs_decay_start=bs_decay_start, bs_decay_rate=bs_decay_rate,
                                        wd_bs_scaling=wd_bs_scaling)
        tqdm.write(f"[objective] Params: lr_base={lr_base:.2e}, lr_peak={lr_peak:.2e}, num_warmup_epochs={num_warmup_epochs}, total_epochs={total_epochs}, bs_decay_start={bs_decay_start}, bs_decay_rate={bs_decay_rate:.3f}, wd_bs_scaling={wd_bs_scaling:.3f}, bs_final={bs_final} => Acc: {acc:.4f}")
        return acc
    
    pbounds = {
        "lr_base": (1e-6, 1e-2),
        "lr_peak": (1e-3, 1e-2),
        "num_warmup_epochs": (1, 4),
        "total_epochs": (10, 30),
        "bs_decay_start": (1, 10),
        "bs_decay_rate": (0.5, 2.0),
        "wd_bs_scaling": (-1.0, 1.0),
        "bs_final": (16, 128)
    }

    optimizer = BayesianOptimization(
        f=objective,
        pbounds=pbounds,
        random_state=42,
    )

    tqdm.write("[bayes_opt] Starting Bayesian Optimization over LR and Batch Scheduler parameters...")
    optimizer.maximize(
        init_points=5,
        n_iter=200,
    )

    best_params = optimizer.max["params"]
    best_params["num_warmup_epochs"] = int(round(best_params["num_warmup_epochs"]))
    best_params["total_epochs"] = int(round(best_params["total_epochs"]))
    best_params["bs_decay_start"] = int(round(best_params["bs_decay_start"]))
    best_acc = optimizer.max["target"]
    tqdm.write(f"[bayes_opt] Best parameters found: {best_params} with test accuracy: {best_acc:.4f}")

    lr_end = 1e-6
    best_lr_schedule = generate_lr_schedule(best_params["lr_base"], best_params["lr_peak"], lr_end,
                                             best_params["num_warmup_epochs"], best_params["total_epochs"])
    constant_wd = 1e-5
    final_acc, final_state = evaluate_schedule_full(best_lr_schedule, constant_wd, init_sd,
                                                    train_tds, val_tds, test_tds,
                                                    total_epochs=best_params["total_epochs"],
                                                    device=DEVICE,
                                                    bs_initial=1028,
                                                    bs_final=best_params["bs_final"],
                                                    bs_decay_start=best_params["bs_decay_start"],
                                                    bs_decay_rate=best_params["bs_decay_rate"],
                                                    wd_bs_scaling=best_params["wd_bs_scaling"])
    meta = {
        "final_acc": final_acc,
        "best_params": best_params,
        "lr_schedule": best_lr_schedule,
        "constant_wd": constant_wd
    }
    save_experiment_results("bayes_opt", meta, final_state, init_sd)
    tqdm.write(f"[main] Final test accuracy with best parameters: {final_acc:.4f}")
    tqdm.write("[main] All done.")

if __name__ == "__main__":
    main()
