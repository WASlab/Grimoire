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
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from model import CNN, EfficientCNN

# You requested to see the cosine warmup + cosine decay schedule implemented.
# We also removed the need for an explicit warmup step count (num_warmup_epochs)
# and removed b1/b2 as tunable parameters altogether.

# ---------------------------
# Global Verbosity (1 = minimal, 2 = maximum)
# ---------------------------
VERBOSE = 1
def vprint(msg, level=1):
    if VERBOSE >= level:
        tqdm.write(msg)

# ---------------------------
# Global Objective Metric
# ---------------------------
# Acceptable values: "accuracy", "macro_f1", "micro_f1", "weighted_f1", "precision", "recall"
OBJECTIVE_METRIC = "accuracy"

# =============== Hyperparameters ===============
NUM_EPOCHS         = 10
BATCH_SIZE         = 1028
DEVICE             = "cuda" if torch.cuda.is_available() else "cpu"
SAVED_TENSORSETS_DIR = "./saved_tensorsets"
RESULTS_DIR        = "./kmnist/results_bayes_opt"
os.makedirs(RESULTS_DIR, exist_ok=True)

# FIXED seed for data loading and model initialization.
DATA_LOADER_SEED   = 42

EARLYSTOP_PATIENCE = float('nan')  # Set to finite value (e.g. 2) to enable early stopping, or NaN to disable
THRESH_ZERO        = 1e-12

MODEL_NAME = "EfficientCNN_v1"  # Global model identifier

CHECKPOINT_META_PATH = os.path.join(RESULTS_DIR, f"bayes_opt_checkpoint_meta_{MODEL_NAME}.json")
CHECKPOINT_OPTIMIZER_PATH = os.path.join(RESULTS_DIR, f"bayes_opt_checkpoint_optimizer_{MODEL_NAME}.pkl")

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
    vprint(f"[load_tensor_datasets] Loading KMNIST TensorDatasets from '{save_dir}'...", level=2)
    # Adjust these paths to the actual files for your environment
    train_tds = torch.load("./kmnist/results_augmentation_search/kmnist_train_augmented_tensorset.pth")
    val_tds = torch.load("./kmnist/saved_tensorsets/kmnist_val_tensorset.pth")
    test_tds = torch.load("./kmnist/saved_tensorsets/kmnist_test_tensorset.pth")
    vprint(f"[load_tensor_datasets] Done. (Train={len(train_tds)}, Val={len(val_tds)}, Test={len(test_tds)})", level=2)
    return train_tds, val_tds, test_tds

def get_dataloaders(train_tds, val_tds, test_tds, batch_size=32, shuffle_seed=DATA_LOADER_SEED):
    vprint(f"[get_dataloaders] Creating DataLoaders (batch_size={batch_size}, seed={shuffle_seed})...", level=2)
    g = torch.Generator()
    g.manual_seed(shuffle_seed)
    train_loader = DataLoader(train_tds, batch_size=batch_size, shuffle=True, generator=g)
    val_loader = DataLoader(val_tds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_tds, batch_size=batch_size, shuffle=False)
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
# Cosine Warmup + Cosine Decay Schedule
# ---------------------------
def generate_cosine_warmup_decay_schedule(lr_base, lr_peak, lr_end, total_epochs):
    """
    Piecewise schedule that:
      - In the first half of the epochs, does a half-cosine from lr_base to lr_peak (warmup).
      - In the second half, does a half-cosine from lr_peak down to lr_end (decay).
    This removes the need for an explicit 'num_warmup_epochs' param.
    """
    schedule = []
    for epoch in range(total_epochs):
        p = epoch / max(1, (total_epochs - 1))  # fraction of progress [0..1]
        if p <= 0.5:
            # Warmup: from lr_base up to lr_peak
            # Map p in [0..0.5] to p2 in [0..1]
            p2 = p / 0.5
            # Half-cosine from lr_base to lr_peak
            lr = lr_base + 0.5 * (lr_peak - lr_base) * (1 - math.cos(math.pi * p2))
        else:
            # Decay: from lr_peak down to lr_end
            # Map p in [0.5..1] to p2 in [0..1]
            p2 = (p - 0.5) / 0.5
            # Half-cosine from lr_peak down to lr_end
            lr = lr_peak + 0.5 * (lr_end - lr_peak) * (1 - math.cos(math.pi * p2))
        schedule.append(lr)
    return schedule

# ---------------------------
# Evaluation Function (Full Training) with Toggleable Early Stopping
# ---------------------------
def evaluate_schedule_full(lr_schedule, wd_schedule, init_state_dict, train_loader, val_loader, test_loader,
                           num_epochs=NUM_EPOCHS, device=DEVICE, weight_decay=None,
                           early_stop_patience=EARLYSTOP_PATIENCE):
    num_classes = int(train_loader.dataset.tensors[1].max().item()+1)
    model = EfficientCNN(num_classes=num_classes).to(device)
    model.load_state_dict(init_state_dict)
    criterion = nn.CrossEntropyLoss()

    # We'll use AdamW with fixed betas
    # (b1 and b2 are removed from tuning and remain as defaults)
    if weight_decay is None:
        weight_decay = wd_schedule[0]
    optimizer = optim.AdamW(model.parameters(), lr=lr_schedule[0], weight_decay=weight_decay,betas=(.80, .999))

    scaler = torch.cuda.amp.GradScaler()

    best_val_loss = float('inf')
    best_state = None
    epochs_no_improve = 0
    prev_val_loss = None

    for e in range(1, num_epochs+1):
        current_lr = lr_schedule[e-1]
        current_wd = wd_schedule[e-1]
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

        val_loss = compute_loss(model, val_loader, criterion, device)
        vprint(f"[early_stop] Epoch {e}: validation loss = {val_loss:.4f}", level=2)

        # Early Stopping logic
        if not math.isnan(early_stop_patience):
            if prev_val_loss is None or val_loss < prev_val_loss - 1e-6:
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
            prev_val_loss = val_loss

            if val_loss < best_val_loss - 1e-6:
                best_val_loss = val_loss
                best_state = copy.deepcopy(model.state_dict())

            if epochs_no_improve >= early_stop_patience:
                vprint(f"[early_stop] Early stopping triggered at epoch {e}.", level=2)
                break
        else:
            if val_loss < best_val_loss - 1e-6:
                best_val_loss = val_loss
                best_state = copy.deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)

    # Evaluate on the test set
    model.eval()
    tot_correct, tot_samples = 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            tot_correct += (preds == labels).sum().item()
            tot_samples += images.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = tot_correct / tot_samples
    metrics = {"accuracy": acc}
    if OBJECTIVE_METRIC in {"macro_f1", "micro_f1", "weighted_f1", "precision", "recall"}:
        metrics["macro_f1"] = f1_score(all_labels, all_preds, average="macro")
        metrics["micro_f1"] = f1_score(all_labels, all_preds, average="micro")
        metrics["weighted_f1"] = f1_score(all_labels, all_preds, average="weighted")
        metrics["precision"] = precision_score(all_labels, all_preds, average="macro")
        metrics["recall"] = recall_score(all_labels, all_preds, average="macro")
    return metrics, copy.deepcopy(model.state_dict())

# ---------------------------
# Global variables for objective function (set in main)
# ---------------------------
TRAIN_LOADER = None
VAL_LOADER = None
TEST_LOADER = None
INIT_SD = None

# ---------------------------
# Objective Function (for pickling)
# ---------------------------
def objective(lr_peak, base_frac, end_frac, weight_decay):
    """
    Now uses a piecewise cosine warmup + cosine decay schedule from:
      lr_base = base_frac * lr_peak
      lr_end  = end_frac * lr_peak
    We removed num_warmup_epochs and b1/b2 from the search.
    """
    lr_base = base_frac * lr_peak
    lr_end = end_frac * lr_peak

    # Generate the schedule
    lr_schedule = generate_cosine_warmup_decay_schedule(lr_base, lr_peak, lr_end, NUM_EPOCHS)
    wd_schedule = [weight_decay]*NUM_EPOCHS

    metrics, _ = evaluate_schedule_full(
        lr_schedule=lr_schedule,
        wd_schedule=wd_schedule,
        init_state_dict=INIT_SD,
        train_loader=TRAIN_LOADER,
        val_loader=VAL_LOADER,
        test_loader=TEST_LOADER,
        num_epochs=NUM_EPOCHS,
        device=DEVICE,
        weight_decay=weight_decay,
        early_stop_patience=EARLYSTOP_PATIENCE
    )
    return metrics.get(OBJECTIVE_METRIC, metrics["accuracy"])

# ---------------------------
# Checkpointing Utilities
# ---------------------------
def save_checkpoint(optimizer_obj, meta, checkpoint_meta_path, checkpoint_optimizer_path):
    with open(checkpoint_optimizer_path, "wb") as f:
        pickle.dump(optimizer_obj, f)
    with open(checkpoint_meta_path, "w") as f:
        json.dump(meta, f, indent=4)
    vprint("[checkpoint] Checkpoint saved.", level=2)

def load_checkpoint(checkpoint_meta_path, checkpoint_optimizer_path):
    if os.path.exists(checkpoint_meta_path) and os.path.exists(checkpoint_optimizer_path):
        with open(checkpoint_meta_path, "r") as f:
            meta = json.load(f)
        if meta.get("model_name", None) != MODEL_NAME:
            vprint("[checkpoint] Model name mismatch. Ignoring checkpoint.", level=2)
            return None, None
        with open(checkpoint_optimizer_path, "rb") as f:
            optimizer_obj = pickle.load(f)
        vprint("[checkpoint] Checkpoint loaded.", level=2)
        return meta, optimizer_obj
    else:
        return None, None

# ---------------------------
# Saving Experiment Results
# ---------------------------
def save_experiment_results(method_name, meta, best_state, init_state_dict):
    base_init_path = os.path.join(RESULTS_DIR, f"problem1_{method_name}_init_state_dict_{MODEL_NAME}")
    base_best_path = os.path.join(RESULTS_DIR, f"problem1_{method_name}_best_state_dict_{MODEL_NAME}")
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
    json_path = os.path.join(RESULTS_DIR, f"problem1_{method_name}_results_{MODEL_NAME}.json")
    with open(json_path, "w") as f:
        json.dump(meta, f, indent=4)
    vprint(f"[{method_name.upper()}] Results saved to {json_path}.", level=2)

# ---------------------------
# Misclassification Report Utility (for final test run)
# ---------------------------
def print_misclassification_report(model, test_loader, device=DEVICE):
    model.eval()
    misclassified = {}
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            for true, pred in zip(labels.cpu().numpy(), preds.cpu().numpy()):
                if true != pred:
                    misclassified[true] = misclassified.get(true, 0) + 1
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    vprint("Misclassification Report:", level=1)
    for cls in sorted(misclassified.keys()):
        vprint(f"Class {cls}: {misclassified[cls]} misclassifications", level=1)

# ---------------------------
# Main Pipeline with Bayesian Optimization, Toggleable Early Stopping, and Checkpointing
# ---------------------------
def main():
    vprint(f"[main] Setting up results folder: '{RESULTS_DIR}'", level=1)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    global GLOBAL_EVAL_COUNT, TRAIN_LOADER, VAL_LOADER, TEST_LOADER, INIT_SD
    GLOBAL_EVAL_COUNT = 0

    # Fix seed for the entire optimization run
    torch.manual_seed(42)
    random.seed(42)

    checkpoint_meta = {"model_name": MODEL_NAME, "completed": "NULL", "best_params": None}

    vprint("[main] Loading dataset + building initial EfficientCNN model.", level=1)
    train_tds, val_tds, test_tds = load_tensor_datasets(SAVED_TENSORSETS_DIR)
    TRAIN_LOADER, VAL_LOADER, TEST_LOADER = get_dataloaders(train_tds, val_tds, test_tds, BATCH_SIZE)
    num_classes = int(TRAIN_LOADER.dataset.tensors[1].max().item() + 1)

    # Fix model initialization seed
    torch.manual_seed(42)
    random.seed(42)
    init_model = EfficientCNN(num_classes=num_classes)
    INIT_SD = copy.deepcopy(init_model.state_dict())

    # Define the parameter bounds (no num_warmup_epochs, no b1/b2)
    pbounds = {
        "lr_peak": (1e-3, 1e-2),
        "base_frac": (0.1, 0.9),
        "end_frac": (0.0, 0.7),
        "weight_decay": (1e-6, 1e-3)
    }

    # Attempt to load from checkpoint
    meta, optimizer_obj = load_checkpoint(CHECKPOINT_META_PATH, CHECKPOINT_OPTIMIZER_PATH)
    if optimizer_obj is None:
        vprint("[bayes_opt] No checkpoint found. Starting fresh Bayesian Optimization.", level=1)
        optimizer_obj = BayesianOptimization(
            f=objective,
            pbounds=pbounds,
            random_state=42,
        )
        meta = {"model_name": MODEL_NAME, "completed": "NULL", "best_params": None}
        optimizer_obj.maximize(init_points=15, n_iter=0)
    else:
        vprint("[bayes_opt] Resuming from checkpoint.", level=1)

    total_iter = 100  # Number of subsequent BO iterations
    checkpoint_interval = 10  # checkpoint every 10 iterations

    try:
        for i in range(0, total_iter, checkpoint_interval):
            iter_this_block = min(checkpoint_interval, total_iter - i)
            vprint(f"[bayes_opt] Running iterations {i+1} to {i+iter_this_block}...", level=1)
            optimizer_obj.maximize(n_iter=iter_this_block)
            meta["best_params"] = optimizer_obj.max.get("params", None)
            meta["completed"] = "NULL"
            save_checkpoint(optimizer_obj, meta, CHECKPOINT_META_PATH, CHECKPOINT_OPTIMIZER_PATH)
    except Exception as e:
        vprint(f"[error] Optimization interrupted: {e}", level=1)
        meta["best_params"] = optimizer_obj.max.get("params", None)
        meta["completed"] = "NULL"
        save_checkpoint(optimizer_obj, meta, CHECKPOINT_META_PATH, CHECKPOINT_OPTIMIZER_PATH)
        raise e

    # Finished all iterations
    meta["completed"] = "FULL"
    meta["best_params"] = optimizer_obj.max["params"]
    best_params = optimizer_obj.max["params"]
    best_acc = optimizer_obj.max["target"]
    vprint(f"[bayes_opt] Best parameters found: {best_params} with test metric: {best_acc:.4f}", level=1)

    # Reconstruct schedule for final evaluation
    lr_peak = best_params["lr_peak"]
    lr_base = best_params["base_frac"] * lr_peak
    lr_end = best_params["end_frac"] * lr_peak
    best_wd = best_params["weight_decay"]

    best_lr_schedule = generate_cosine_warmup_decay_schedule(lr_base, lr_peak, lr_end, NUM_EPOCHS)
    best_wd_schedule = [best_wd] * NUM_EPOCHS

    final_metrics, final_state = evaluate_schedule_full(
        lr_schedule=best_lr_schedule,
        wd_schedule=best_wd_schedule,
        init_state_dict=INIT_SD,
        train_loader=TRAIN_LOADER,
        val_loader=VAL_LOADER,
        test_loader=TEST_LOADER,
        num_epochs=NUM_EPOCHS,
        device=DEVICE,
        weight_decay=best_wd,
        early_stop_patience=EARLYSTOP_PATIENCE
    )
    final_acc = final_metrics.get("accuracy", final_metrics["accuracy"])
    meta["final_acc"] = final_acc
    meta["lr_schedule"] = best_lr_schedule
    meta["wd_schedule"] = best_wd_schedule

    save_experiment_results("bayes_opt", meta, final_state, INIT_SD)

    # Remove checkpoint files once we've finalized
    if os.path.exists(CHECKPOINT_META_PATH):
        os.remove(CHECKPOINT_META_PATH)
    if os.path.exists(CHECKPOINT_OPTIMIZER_PATH):
        os.remove(CHECKPOINT_OPTIMIZER_PATH)

    vprint(f"[main] Final test accuracy with best parameters: {final_acc:.4f}", level=1)
    vprint("[main] All done.", level=1)


if __name__ == "__main__":
    main()
