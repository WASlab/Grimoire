"""
File: optimize_modular.py

This script demonstrates a more modular approach to Bayesian Optimization of hyperparameters,
with flexible metrics, an optional test-based objective, and an epoch-based LR scheduler.

Key Features:
  - Multiple metrics (accuracy, precision, recall, macro_f1, micro_f1, weighted_f1)
  - Default optimization on validation metrics (test optional)
  - Integration with `EpochLRScheduler` from `grimoire.main.schedulers`
  - Checkpointing, early-stopping, and partial runs
"""

import os
import copy
import json
import pickle
import random
import math
import warnings

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from bayes_opt import BayesianOptimization

# For multiple metrics
from sklearn.metrics import (f1_score, precision_score, recall_score, accuracy_score)

# Example of an external LR scheduler you'd import
# from grimoire.main.schedulers import EpochLRScheduler

########################################
# Global Config & Utility
########################################

VERBOSE = 1
def vprint(msg, level=1):
    """Simple verbosity-controlled print."""
    if VERBOSE >= level:
        tqdm.write(msg)

# Available metrics
VALID_METRICS = {"accuracy", "precision", "recall", "macro_f1", "micro_f1", "weighted_f1"}

# Default metric & usage
OBJECTIVE_METRIC = "accuracy"  # e.g. "accuracy", "macro_f1", ...
OBJECTIVE_ON = "val"           # "val" or "test" for objective scoring
NUM_EPOCHS = 10
BATCH_SIZE = 1028
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

RESULTS_DIR = "./results_modular"
os.makedirs(RESULTS_DIR, exist_ok=True)

DATA_LOADER_SEED = 42
EARLYSTOP_PATIENCE = float('nan')  # e.g. 2 for patience, or nan to disable
MODEL_NAME = "EfficientCNN_v1"

CHECKPOINT_META_PATH = os.path.join(RESULTS_DIR, f"bayes_opt_checkpoint_meta_{MODEL_NAME}.json")
CHECKPOINT_OPTIMIZER_PATH = os.path.join(RESULTS_DIR, f"bayes_opt_checkpoint_optimizer_{MODEL_NAME}.pkl")


########################################
# 1) Dataset Loading & Dataloaders
########################################
def load_tensor_datasets(train_path, val_path, test_path):
    """Loads TensorDatasets from given file paths."""
    vprint(f"[load_tensor_datasets] Loading from paths:\n  train={train_path}\n  val={val_path}\n  test={test_path}", level=2)
    train_tds = torch.load(train_path)
    val_tds   = torch.load(val_path)
    test_tds  = torch.load(test_path)
    vprint(f"[load_tensor_datasets] Loaded: Train={len(train_tds)}, Val={len(val_tds)}, Test={len(test_tds)}", level=2)
    return train_tds, val_tds, test_tds

def get_dataloaders(train_tds, val_tds, test_tds, batch_size=BATCH_SIZE, shuffle_seed=DATA_LOADER_SEED):
    """Creates train/val/test DataLoaders from TensorDatasets."""
    vprint(f"[get_dataloaders] Creating DataLoaders (batch_size={batch_size}, seed={shuffle_seed})", level=2)
    g = torch.Generator()
    g.manual_seed(shuffle_seed)
    train_loader = DataLoader(train_tds, batch_size=batch_size, shuffle=True, generator=g)
    val_loader   = DataLoader(val_tds,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_tds,  batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader

def compute_loss(model, loader, criterion, device=DEVICE):
    """Computes average loss on a given loader."""
    model.eval()
    total_loss, total_samples = 0.0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            total_samples += images.size(0)
    if total_samples == 0:
        return float('inf')
    return total_loss / total_samples


########################################
# 2) Metric Computation
########################################
def compute_metrics(model, loader, device):
    """
    Computes multiple classification metrics on a given loader.
    Returns a dict of metric_name -> value.
    """
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    micro_f1 = f1_score(all_labels, all_preds, average="micro")
    weighted_f1 = f1_score(all_labels, all_preds, average="weighted")
    prec = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    rec  = recall_score(all_labels, all_preds, average="macro", zero_division=0)

    metrics_dict = {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
        "weighted_f1": weighted_f1,
        "precision": prec,
        "recall": rec
    }
    return metrics_dict


########################################
# 3) LR Scheduler (Epoch-based) [Example]
#    You can import from grimoire.main.schedulers if you prefer
########################################
class EpochLRScheduler:
    """
    An epoch-wise learning rate scheduler that updates on *every epoch*.
    This is a demonstration example. You may import from grimoire.main.schedulers instead.
    """
    def __init__(
        self,
        optimizer,
        total_epochs,
        warmup_epochs=0,
        init_lr=1e-4,
        peak_lr=1e-3,
        end_lr=1e-5,
        warmup_type="linear",
        decay_type="cosine",
    ):
        self.optimizer = optimizer
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs
        self.init_lr = init_lr
        self.peak_lr = peak_lr
        self.end_lr = end_lr
        self.warmup_type = warmup_type.lower()
        self.decay_type = decay_type.lower()

        self.current_epoch = 0

        # Basic validation
        if self.init_lr is False and self.warmup_epochs > 0:
            warnings.warn("warmup_epochs>0 but init_lr=False => No warmup.")
        if self.end_lr is False and (self.decay_type != "static"):
            warnings.warn("end_lr=False but decay_type!=static => no decay, stay at peak_lr.")
        if self.total_epochs <= self.warmup_epochs and self.warmup_epochs > 0:
            warnings.warn("No decay phase will occur. total_epochs<=warmup_epochs")

        # Initialize LR
        starting_lr = self.init_lr if self.init_lr is not False else self.peak_lr
        self._set_lr(starting_lr)

    def step(self):
        """Call once per epoch (at the end of epoch)."""
        self.current_epoch += 1
        new_lr = self._compute_lr()
        self._set_lr(new_lr)

    def _compute_lr(self):
        """Computes LR for the current epoch, 1-based index."""
        if (self.init_lr is False) and (self.end_lr is False):
            # purely static at peak
            return self.peak_lr

        # Warmup
        if self.init_lr is not False and (self.current_epoch <= self.warmup_epochs):
            return self._warmup_lr()
        else:
            # Decay or stay
            if self.end_lr is False:
                return self.peak_lr
            return self._decay_lr()

    def _warmup_lr(self):
        """Compute LR during warmup phase."""
        progress = self.current_epoch / max(1, self.warmup_epochs)
        if self.warmup_type == "linear":
            return self.init_lr + (self.peak_lr - self.init_lr) * progress
        elif self.warmup_type == "exponential":
            ratio = self.peak_lr / max(1e-12, self.init_lr)
            return self.init_lr * (ratio ** progress)
        elif self.warmup_type == "polynomial":
            return self.init_lr + (self.peak_lr - self.init_lr) * (progress ** 2)
        elif self.warmup_type == "static":
            return self.init_lr
        else:
            raise ValueError(f"Unknown warmup_type={self.warmup_type}")

    def _decay_lr(self):
        """Compute LR during decay phase."""
        decay_epoch = self.current_epoch - self.warmup_epochs
        decay_total = max(1, self.total_epochs - self.warmup_epochs)
        progress = decay_epoch / decay_total

        if self.decay_type == "linear":
            return self.peak_lr + (self.end_lr - self.peak_lr) * progress
        elif self.decay_type == "exponential":
            ratio = self.end_lr / max(1e-12, self.peak_lr)
            return self.peak_lr * (ratio ** progress)
        elif self.decay_type == "polynomial":
            return self.peak_lr + (self.end_lr - self.peak_lr) * (progress ** 2)
        elif self.decay_type == "cosine":
            return self.end_lr + 0.5 * (self.peak_lr - self.end_lr) * (
                1 + math.cos(math.pi * progress)
            )
        elif self.decay_type == "static":
            return self.peak_lr
        else:
            raise ValueError(f"Unknown decay_type={self.decay_type}")

    def _set_lr(self, lr):
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def get_lr(self):
        """Return current LR."""
        return self.optimizer.param_groups[0]["lr"]


########################################
# 4) "Full" Training + (Optionally) Early-Stopping
#    + Example usage of the above epoch-based scheduler
########################################
def train_and_evaluate(
    model,
    train_loader,
    val_loader,
    test_loader,
    objective_metric="accuracy",
    objective_on="val",
    device=DEVICE,
    num_epochs=10,
    early_stop_patience=float("nan"),
    lr_scheduler_params=None
):
    """
    Trains model for num_epochs, optionally uses epoch-based scheduling,
    does optional early stopping, and returns final metrics on both val and test.

    - objective_metric can be "accuracy", "precision", "recall", "macro_f1", "micro_f1", "weighted_f1".
    - objective_on can be "val" (default) or "test". 
      If "test", we measure the objective on the test set each epoch. 
      If "val", we measure on the val set each epoch (typical approach).
    - lr_scheduler_params example:
          {
            "warmup_epochs": 2,
            "init_lr": 1e-4,
            "peak_lr": 1e-3,
            "end_lr": 1e-5,
            "warmup_type": "linear",
            "decay_type": "cosine",
          }
      If None, no scheduling is used.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)  # or pass externally

    # Build a scheduler if provided
    if lr_scheduler_params is not None:
        total_epochs = lr_scheduler_params.get("total_epochs", num_epochs)
        scheduler = EpochLRScheduler(
            optimizer=optimizer,
            total_epochs=total_epochs,
            warmup_epochs=lr_scheduler_params.get("warmup_epochs", 0),
            init_lr=lr_scheduler_params.get("init_lr", 1e-4),
            peak_lr=lr_scheduler_params.get("peak_lr", 1e-3),
            end_lr=lr_scheduler_params.get("end_lr", 1e-5),
            warmup_type=lr_scheduler_params.get("warmup_type", "linear"),
            decay_type=lr_scheduler_params.get("decay_type", "cosine"),
        )
    else:
        scheduler = None

    best_score = -float("inf")
    best_state = None
    epochs_no_improve = 0
    measure_loader = val_loader if objective_on == "val" else test_loader

    for epoch in range(1, num_epochs+1):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Step scheduler
        if scheduler is not None:
            scheduler.step()

        # Evaluate on val or test loader
        metrics_dict = compute_metrics(model, measure_loader, device)
        current_score = metrics_dict.get(objective_metric, 0.0)
        vprint(f"[Epoch {epoch}] {objective_on} {objective_metric} = {current_score:.4f}", level=2)

        # Early stopping logic
        if not math.isnan(early_stop_patience):
            if current_score > best_score + 1e-9:
                best_score = current_score
                best_state = copy.deepcopy(model.state_dict())
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= early_stop_patience:
                vprint(f"[train_and_evaluate] Early stopping at epoch {epoch}. Best {objective_on} {objective_metric} = {best_score:.4f}", 2)
                break
        else:
            # no early stopping, still track best
            if current_score > best_score:
                best_score = current_score
                best_state = copy.deepcopy(model.state_dict())

    # Load best state
    if best_state is not None:
        model.load_state_dict(best_state)

    # Evaluate final val & test metrics
    val_metrics = compute_metrics(model, val_loader, device) if val_loader is not None else {}
    test_metrics = compute_metrics(model, test_loader, device) if test_loader is not None else {}

    return val_metrics, test_metrics, copy.deepcopy(model.state_dict())


########################################
# 5) Example Bayesian Optimization Workflow
########################################
def objective_function(**kwargs):
    """
    A sample objective function that uses 'train_and_evaluate' with given hyperparameters.
    Returns the chosen metric from either val or test.

    Expect 'kwargs' to contain your tuning hyperparams (like lr, weight_decay, etc.).
    """
    # For demonstration, we'll do something naive.
    # In reality, you'd pass these hyperparams to your model or optimizer or scheduler.
    # We'll just produce a random number here or call the train method.

    # e.g. let's pretend we build a new model each time
    # model = MyNetwork(...).to(DEVICE)

    # or we do something simpler for demonstration:
    dummy_score = random.random()
    return dummy_score


def run_bayes_opt():
    """
    Illustrates how you'd run BayesianOptimization with a modular approach,
    letting 'objective_function' define the training steps.
    """
    pbounds = {
        "lr_peak": (1e-3, 1e-2),
        "my_other_param": (0.0, 1.0),
    }
    bo = BayesianOptimization(f=objective_function, pbounds=pbounds, random_state=42)
    bo.maximize(init_points=5, n_iter=20)
    vprint(f"Best found: {bo.max['params']} => {bo.max['target']:.4f}")
    return bo


########################################
# 6) MAIN CLI or callable pipeline
########################################
def main():
    """Example main function showing usage of the above components."""
    vprint("[main] Starting a more modular Bayesian Optimization pipeline...", level=1)

    # 1) Load data
    train_path = r"kmnist\results_augmentation_search2\kmnist_train_augmented_tensorset.pth"
    val_path   = r"kmnist\saved_tensorsets\kmnist_val_tensorset.pth"
    test_path  = r"kmnist\saved_tensorsets\kmnist_test_tensorset.pth"
    train_tds, val_tds, test_tds = load_tensor_datasets(train_path, val_path, test_path)
    train_loader, val_loader, test_loader = get_dataloaders(train_tds, val_tds, test_tds, BATCH_SIZE)

    # 2) Example usage of an epoch-based LR scheduler in final training
    #    (If you want, you can also incorporate it into the BO step.)
    lr_scheduler_params = {
        "warmup_epochs": 2,
        "init_lr": 1e-4,
        "peak_lr": 1e-3,
        "end_lr": 1e-5,
        "warmup_type": "cosine",
        "decay_type": "cosine",
    }

    # 3) Build model, do final training
    from model import EfficientCNN
    model = EfficientCNN(num_classes=10).to(DEVICE)

    # 4) Train & Evaluate (default objective=val accuracy)
    val_metrics, test_metrics, best_state = train_and_evaluate(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        objective_metric="accuracy",    # e.g. "macro_f1" ...
        objective_on="val",            # or "test"
        device=DEVICE,
        num_epochs=NUM_EPOCHS,
        early_stop_patience=2,
        lr_scheduler_params=lr_scheduler_params
    )

    vprint(f"[main] Final Val Metrics: {val_metrics}", 1)
    vprint(f"[main] Final Test Metrics: {test_metrics}", 1)

    # 5) If you want to run your Bayesian Optimization
    bo = run_bayes_opt()
    vprint("[main] Done with the example pipeline!", 1)


if __name__ == "__main__":
    main()
