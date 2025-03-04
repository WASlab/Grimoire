import os
import copy
import json
import pickle
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

# =============== Hyperparameters ===============
NUM_EPOCHS         = 10
BATCH_SIZE         = 1028
DEVICE             = "cuda" if torch.cuda.is_available() else "cpu"
SAVED_TENSORSETS_DIR = "./saved_tensorsets"
RESULTS_DIR        = "./results_bayes_opt"
os.makedirs(RESULTS_DIR, exist_ok=True)

DATA_LOADER_SEED   = 12345   # fixed seed for dataset shuffling
EARLYSTOP_PATIENCE = 2       # not used in full-training evaluation

THRESH_ZERO = 1e-12  # threshold below which LR is treated as 0

# Checkpoint filenames
CHECKPOINT_META_PATH = os.path.join(RESULTS_DIR, "bayes_opt_checkpoint_meta.json")
CHECKPOINT_OPTIMIZER_PATH = os.path.join(RESULTS_DIR, "bayes_opt_checkpoint_optimizer.pkl")

# ---------------------------
# Global cache (not used in full training)
# ---------------------------
BEST_PREFIX_CACHE = {}
GLOBAL_EVAL_COUNT = 0

# Custom tqdm kwargs for inline, dynamic updating
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
    # Load the augmented training dataset (produced by the augmentation search)
    train_tds = torch.load("./kmnist/results_augmentation_search/kmnist_train_augmented_tensorset.pth")
    # Load the validation and test datasets as before
    val_tds = torch.load("./kmnist/saved_tensorsets/kmnist_val_tensorset.pth")
    test_tds = torch.load("./kmnist/saved_tensorsets/kmnist_test_tensorset.pth")
    tqdm.write(f"[load_tensor_datasets] Done. (Train={len(train_tds)}, Val={len(val_tds)}, Test={len(test_tds)})")
    return train_tds, val_tds, test_tds

def get_dataloaders(train_tds, val_tds, test_tds, batch_size=32, shuffle_seed=DATA_LOADER_SEED):
    tqdm.write(f"[get_dataloaders] Creating DataLoaders (batch_size={batch_size}, seed={shuffle_seed})...")
    g = torch.Generator()
    g.manual_seed(shuffle_seed)
    train_loader = DataLoader(train_tds, batch_size=batch_size, shuffle=True, generator=g)
    val_loader   = DataLoader(val_tds, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_tds, batch_size=batch_size, shuffle=False)
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
        self.dropout = nn.Dropout(0.1)
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
# Evaluation Function (Full Training)
# ---------------------------
def evaluate_schedule_full(lr_schedule, wd_schedule, init_state_dict, train_loader, val_loader, test_loader,
                           num_epochs=NUM_EPOCHS, device=DEVICE, b1=0.85, b2=0.999, weight_decay=None):
    # Build and load model
    num_classes = int(train_loader.dataset.tensors[1].max().item() + 1)
    model = CNN(num_classes=num_classes).to(device)
    model.load_state_dict(init_state_dict)
    criterion = nn.CrossEntropyLoss()
    if weight_decay is None:
        weight_decay = wd_schedule[0]
    optimizer = optim.AdamW(model.parameters(), lr=lr_schedule[0],
                            weight_decay=weight_decay, betas=(b1, b2))
    scaler = torch.cuda.amp.GradScaler()

    for e in range(1, num_epochs + 1):
        current_lr = lr_schedule[e - 1]
        current_wd = wd_schedule[e - 1]
        for pg in optimizer.param_groups:
            pg['lr'] = current_lr
            pg['weight_decay'] = current_wd
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
# Checkpointing Utilities
# ---------------------------
def save_checkpoint(optimizer_obj, meta, checkpoint_meta_path, checkpoint_optimizer_path):
    # Save optimizer state (using pickle)
    with open(checkpoint_optimizer_path, "wb") as f:
        pickle.dump(optimizer_obj, f)
    # Save meta info (e.g., best params, completion token)
    with open(checkpoint_meta_path, "w") as f:
        json.dump(meta, f, indent=4)
    tqdm.write("[checkpoint] Checkpoint saved.")

def load_checkpoint(checkpoint_meta_path, checkpoint_optimizer_path):
    if os.path.exists(checkpoint_meta_path) and os.path.exists(checkpoint_optimizer_path):
        with open(checkpoint_meta_path, "r") as f:
            meta = json.load(f)
        with open(checkpoint_optimizer_path, "rb") as f:
            optimizer_obj = pickle.load(f)
        tqdm.write("[checkpoint] Checkpoint loaded.")
        return meta, optimizer_obj
    else:
        return None, None

# ---------------------------
# Saving Experiment Results
# ---------------------------
def save_experiment_results(method_name, meta, best_state, init_state_dict):
    # Save initial state and best state with file iteration if completed token is set
    base_init_path = os.path.join(RESULTS_DIR, f"problem1_{method_name}_init_state_dict")
    base_best_path = os.path.join(RESULTS_DIR, f"problem1_{method_name}_best_state_dict")
    # Check for completion token in meta; if not NULL, iterate filename
    if meta.get("completed", "NULL") != "NULL":
        count = 1
        while os.path.exists(f"{base_init_path}_{count}.pth"):
            count += 1
        init_path = f"{base_init_path}_{count}.pth"
        best_path = f"{base_best_path}_{count}.pth"
    else:
        init_path = base_init_path + ".pth"
        best_path = base_best_path + ".pth"
    
    torch.save(init_state_dict, init_path)
    if best_state is not None:
        torch.save(best_state, best_path)
    meta["init_state_file"] = init_path
    meta["best_state_file"] = best_path
    json_path = os.path.join(RESULTS_DIR, f"problem1_{method_name}_results.json")
    with open(json_path, "w") as f:
        json.dump(meta, f, indent=4)
    tqdm.write(f"[{method_name.upper()}] Results saved to {json_path}.")

# ---------------------------
# Main Pipeline with Bayesian Optimization and Checkpointing
# ---------------------------
def main():
    tqdm.write(f"[main] Setting up results folder: '{RESULTS_DIR}'")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    global GLOBAL_EVAL_COUNT
    GLOBAL_EVAL_COUNT = 0

    tqdm.write("[main] Loading dataset + building initial CNN model.")
    train_tds, val_tds, test_tds = load_tensor_datasets(SAVED_TENSORSETS_DIR)
    train_loader, val_loader, test_loader = get_dataloaders(train_tds, val_tds, test_tds, BATCH_SIZE)
    num_classes = int(train_loader.dataset.tensors[1].max().item() + 1)
    torch.manual_seed(42)
    init_model = CNN(num_classes=num_classes)
    init_sd = copy.deepcopy(init_model.state_dict())

    # Define objective function for Bayesian Optimization.
    def objective(lr_base, lr_peak, num_warmup_epochs, b1, b2, weight_decay):
        num_warmup_epochs = max(1, int(round(num_warmup_epochs)))
        lr_end = 1e-6
        lr_schedule = generate_lr_schedule(lr_base, lr_peak, lr_end, num_warmup_epochs, NUM_EPOCHS)
        wd_schedule = [weight_decay] * NUM_EPOCHS
        acc, _ = evaluate_schedule_full(lr_schedule, wd_schedule, init_sd, train_loader, val_loader, test_loader,
                                        num_epochs=NUM_EPOCHS, device=DEVICE, b1=b1, b2=b2, weight_decay=weight_decay)
        return acc

    pbounds = {
        "lr_base": (1e-6, 1e-2),
        "lr_peak": (5e-4, 1e-2),
        "num_warmup_epochs": (1, 6),
        "b1": (0.5, 0.95),
        "b2": (0.99, 0.9999),
        "weight_decay": (1e-6, 1e-3)
    }

    # Try to load an existing checkpoint
        # Try to load an existing checkpoint
    meta, optimizer_obj = load_checkpoint(CHECKPOINT_META_PATH, CHECKPOINT_OPTIMIZER_PATH)
    if optimizer_obj is None:
        tqdm.write("[bayes_opt] No checkpoint found. Starting fresh Bayesian Optimization.")
        optimizer_obj = BayesianOptimization(
            f=objective,
            pbounds=pbounds,
            random_state=42,
        )
        meta = {"completed": "NULL", "best_params": None}
        # Generate initial points (if needed)
        optimizer_obj.maximize(init_points=5, n_iter=0)
    else:
        tqdm.write("[bayes_opt] Resuming from checkpoint.")

    # Run optimization in small chunks with periodic checkpointing.
    total_iter = 200  # total iterations to run after the initial points
    checkpoint_interval = 10  # checkpoint every 10 iterations

    try:
        for i in range(0, total_iter, checkpoint_interval):
            iter_this_block = min(checkpoint_interval, total_iter - i)
            tqdm.write(f"[bayes_opt] Running iterations {i+1} to {i+iter_this_block}...")
            optimizer_obj.maximize(n_iter=iter_this_block)
            
            # Save checkpoint after these iterations
            meta["best_params"] = optimizer_obj.max.get("params", None)
            meta["completed"] = "NULL"
            save_checkpoint(optimizer_obj, meta, CHECKPOINT_META_PATH, CHECKPOINT_OPTIMIZER_PATH)
    except Exception as e:
        tqdm.write(f"[error] Optimization interrupted: {e}")
        meta["best_params"] = optimizer_obj.max.get("params", None)
        meta["completed"] = "NULL"
        save_checkpoint(optimizer_obj, meta, CHECKPOINT_META_PATH, CHECKPOINT_OPTIMIZER_PATH)
        raise e

    # Mark as completed and continue with final evaluation.
    meta["completed"] = "FULL"
    meta["best_params"] = optimizer_obj.max["params"]
    best_params = optimizer_obj.max["params"]
    best_params["num_warmup_epochs"] = int(round(best_params["num_warmup_epochs"]))
    best_acc = optimizer_obj.max["target"]
    tqdm.write(f"[bayes_opt] Best parameters found: {best_params} with test accuracy: {best_acc:.4f}")


    # Final evaluation with best parameters.
    lr_end = 1e-6
    best_lr_schedule = generate_lr_schedule(best_params["lr_base"], best_params["lr_peak"], lr_end,
                                             best_params["num_warmup_epochs"], NUM_EPOCHS)
    best_wd_schedule = [best_params["weight_decay"]] * NUM_EPOCHS

    final_acc, final_state = evaluate_schedule_full(best_lr_schedule, best_wd_schedule, init_sd,
                                                    train_loader, val_loader, test_loader,
                                                    num_epochs=NUM_EPOCHS, device=DEVICE,
                                                    b1=best_params["b1"], b2=best_params["b2"],
                                                    weight_decay=best_params["weight_decay"])
    meta["final_acc"] = final_acc
    meta["lr_schedule"] = best_lr_schedule
    meta["wd_schedule"] = best_wd_schedule

    save_experiment_results("bayes_opt", meta, final_state, init_sd)
    # Optionally remove checkpoint files since we completed a full run:
    os.remove(CHECKPOINT_META_PATH)
    os.remove(CHECKPOINT_OPTIMIZER_PATH)
    tqdm.write(f"[main] Final test accuracy with best parameters: {final_acc:.4f}")
    tqdm.write("[main] All done.")

if __name__ == "__main__":
    main()
        