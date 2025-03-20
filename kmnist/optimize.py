"""
File: optimize_modular.py

This script demonstrates a more modular approach to Bayesian Optimization of hyperparameters,
with flexible metrics, an epoch-based training routine that uses a step-wise LR scheduler,
and tuning of warmup steps and learning rates (as fractions of the peak LR).

Key Features:
  - Multiple metrics: accuracy, precision, recall, macro_f1, micro_f1, weighted_f1.
  - Default optimization on validation metrics.
  - Integration with a step-wise LR scheduler imported from schedulers.py.
  - Hyperparameters tuned: peak_lr, init_lr_frac, end_lr_frac, warmup_steps_frac, and weight_decay.
  - Total training steps are derived from the dataloader length.
  - Checkpointing, early-stopping, and partial runs can be added.
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
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

# Import the step-wise LR scheduler from your external library


########################################
# Global Config & Utility Functions
########################################

VERBOSE = 1
def vprint(msg, level=1):
    if VERBOSE >= level:
        tqdm.write(msg)

# Valid metrics list
VALID_METRICS = {"accuracy", "precision", "recall", "macro_f1", "micro_f1", "weighted_f1"}

# Default objective settings
OBJECTIVE_METRIC = "accuracy"   # Optimize based on validation accuracy
OBJECTIVE_ON = "val"            # Use validation set during optimization

NUM_EPOCHS = 10               # Proxy training epochs
BATCH_SIZE = 1028
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

RESULTS_DIR = "./results_modular"
os.makedirs(RESULTS_DIR, exist_ok=True)

DATA_LOADER_SEED = 42
EARLYSTOP_PATIENCE = float('nan')  # Set to a finite number to enable early stopping
MODEL_NAME = "EfficientCNNwAttn_v1"  # Example model identifier

# Checkpoint file paths
CHECKPOINT_META_PATH = os.path.join(RESULTS_DIR, f"bayes_opt_checkpoint_meta_{MODEL_NAME}.json")
CHECKPOINT_OPTIMIZER_PATH = os.path.join(RESULTS_DIR, f"bayes_opt_checkpoint_optimizer_{MODEL_NAME}.pkl")
class StepLRScheduler:
    """
    A step-wise learning rate scheduler that updates on *every batch* (each training step).
    
    Features:
      - Optional warmup from init_lr to peak_lr (over warmup_steps).
      - Optional decay from peak_lr to end_lr (over total_steps - warmup_steps).
      - Multiple warmup/decay types: "linear", "exponential", "polynomial", "cosine", "sigmoid", "static", and (for decay) "logarithmic".
      - If init_lr is False, the warmup phase is skipped (immediately using peak_lr).
      - If end_lr is False, the decay phase is skipped (staying at peak_lr after warmup).
      
    Usage:
      1) Initialize with an optimizer, total_steps, warmup_steps, etc.
      2) Call .step() each time you process a batch (i.e., each training step).
      3) The scheduler automatically updates optimizer.param_groups[...]["lr"].
    """

    def __init__(
        self,
        optimizer,
        total_steps,
        warmup_steps=0,
        init_lr=1e-4,
        peak_lr=1e-3,
        end_lr=1e-5,
        warmup_type="linear",
        decay_type="cosine",
    ):
        """
        Args:
            optimizer: torch Optimizer object.
            total_steps (int): total number of steps (batches) for training.
            warmup_steps (int): number of steps used for warmup.
            init_lr (float or bool): starting LR. If False, no warmup phase (start at peak_lr).
            peak_lr (float): LR at the end of warmup. If no warmup, remains constant until decay starts.
            end_lr (float or bool): final LR after decay. If False, no decay occurs; remain at peak_lr.
            warmup_type (str): "linear", "exponential", "polynomial", "cosine", "sigmoid", or "static".
            decay_type (str): "linear", "exponential", "polynomial", "cosine", "sigmoid", "logarithmic", or "static".
        """
        self.optimizer = optimizer
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.init_lr = init_lr
        self.peak_lr = peak_lr
        self.end_lr = end_lr
        self.warmup_type = warmup_type.lower()
        self.decay_type = decay_type.lower()

        self.current_step = 0

        # Validate arguments
        if self.init_lr is False and self.warmup_steps > 0:
            warnings.warn(
                "warmup_steps > 0 but init_lr=False. Warmup phase is disabled; training starts at peak_lr."
            )
        if self.end_lr is False and (self.decay_type != "static"):
            warnings.warn(
                "end_lr=False but decay_type is not 'static'. Decay phase is disabled; LR stays at peak_lr."
            )
        if self.total_steps <= self.warmup_steps and self.warmup_steps > 0:
            warnings.warn(
                f"total_steps={self.total_steps} is less than or equal to warmup_steps={self.warmup_steps}. "
                "Decay phase will not occur."
            )

        # Initialize optimizer with starting LR
        initial = self.init_lr if self.init_lr is not False else self.peak_lr
        self._set_lr(initial)

    def step(self):
        """Call this once per training step (batch)."""
        self.current_step += 1
        new_lr = self._compute_lr()
        self._set_lr(new_lr)

    def _compute_lr(self):
        """Compute the learning rate based on the current step."""
        if (self.init_lr is False) and (self.end_lr is False):
            return self.peak_lr

        if self.init_lr is not False and (self.current_step <= self.warmup_steps):
            return self._warmup_lr()
        else:
            if self.end_lr is False:
                return self.peak_lr
            return self._decay_lr()

    def _warmup_lr(self):
        """Compute LR during the warmup phase."""
        progress = self.current_step / max(1, self.warmup_steps)  # in [0,1]
        if self.warmup_type == "linear":
            return self.init_lr + (self.peak_lr - self.init_lr) * progress
        elif self.warmup_type == "exponential":
            ratio = self.peak_lr / max(1e-12, self.init_lr)
            return self.init_lr * (ratio ** progress)
        elif self.warmup_type == "polynomial":
            return self.init_lr + (self.peak_lr - self.init_lr) * (progress ** 2)
        elif self.warmup_type == "static":
            return self.init_lr
        elif self.warmup_type == "cosine":
            # Cosine warmup: f(0)=0, f(1)=1 using 1-cos((pi/2)*progress)
            return self.init_lr + (self.peak_lr - self.init_lr) * (1 - math.cos((math.pi * progress) / 2))
        elif self.warmup_type == "sigmoid":
            # Sigmoid warmup: Normalize a sigmoid so that f(0)=0 and f(1)=1.
            k = 12.0  # steepness constant
            raw0 = 1 / (1 + math.exp(k * 0.5))
            raw1 = 1 / (1 + math.exp(-k * 0.5))
            raw = 1 / (1 + math.exp(-k * (progress - 0.5)))
            norm_factor = (raw - raw0) / (raw1 - raw0)
            return self.init_lr + (self.peak_lr - self.init_lr) * norm_factor
        else:
            raise ValueError(f"Unknown warmup_type: {self.warmup_type}")

    def _decay_lr(self):
        """Compute LR during the decay phase."""
        decay_step = self.current_step - self.warmup_steps
        decay_total = max(1, self.total_steps - self.warmup_steps)
        progress = decay_step / decay_total  # in [0,1]
        if self.decay_type == "linear":
            return self.peak_lr + (self.end_lr - self.peak_lr) * progress
        elif self.decay_type == "exponential":
            ratio = self.end_lr / max(1e-12, self.peak_lr)
            return self.peak_lr * (ratio ** progress)
        elif self.decay_type == "polynomial":
            return self.peak_lr + (self.end_lr - self.peak_lr) * (progress ** 2)
        elif self.decay_type == "cosine":
            return self.end_lr + 0.5 * (self.peak_lr - self.end_lr) * (1 + math.cos(math.pi * progress))
        elif self.decay_type == "static":
            return self.peak_lr
        elif self.decay_type == "sigmoid":
            # Sigmoid decay: Normalize a sigmoid so that f(0)=0 and f(1)=1.
            k = 12.0
            raw0 = 1 / (1 + math.exp(-k * 0.5))
            raw1 = 1 / (1 + math.exp(k * 0.5))
            raw = 1 / (1 + math.exp(k * (progress - 0.5)))
            norm_factor = (raw - raw0) / (raw1 - raw0)
            return self.peak_lr + (self.end_lr - self.peak_lr) * norm_factor
        elif self.decay_type == "logarithmic":
            # Logarithmic decay: f(progress)=1 - log(1+progress*(e-1)) (normalized so that f(0)=1 and f(1)=0)
            factor = 1 - math.log(1 + progress * (math.e - 1))
            return self.peak_lr * factor + self.end_lr * (1 - factor)
        else:
            raise ValueError(f"Unknown decay_type: {self.decay_type}")

    def _set_lr(self, lr):
        """Sets the learning rate in all parameter groups."""
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def get_lr(self):
        """Returns the current LR from the first parameter group."""
        return self.optimizer.param_groups[0]["lr"]
########################################
# 1) Dataset Loading & Dataloaders
########################################
def load_tensor_datasets(train_path, val_path, test_path):
    vprint(f"[load_tensor_datasets] Loading from:\n  train={train_path}\n  val={val_path}\n  test={test_path}", level=2)
    train_tds = torch.load(train_path)
    val_tds   = torch.load(val_path)
    test_tds  = torch.load(test_path)
    vprint(f"[load_tensor_datasets] Loaded: Train={len(train_tds)}, Val={len(val_tds)}, Test={len(test_tds)}", level=2)
    return train_tds, val_tds, test_tds

def get_dataloaders(train_tds, val_tds, test_tds, batch_size=BATCH_SIZE, shuffle_seed=DATA_LOADER_SEED):
    vprint(f"[get_dataloaders] Creating DataLoaders (batch_size={batch_size}, seed={shuffle_seed})", level=2)
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

########################################
# 2) Metric Computation
########################################
def compute_metrics(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    metrics_dict = {
        "accuracy": accuracy_score(all_labels, all_preds),
        "macro_f1": f1_score(all_labels, all_preds, average="macro"),
        "micro_f1": f1_score(all_labels, all_preds, average="micro"),
        "weighted_f1": f1_score(all_labels, all_preds, average="weighted"),
        "precision": precision_score(all_labels, all_preds, average="macro", zero_division=0),
        "recall": recall_score(all_labels, all_preds, average="macro", zero_division=0)
    }
    return metrics_dict

########################################
# 3) Training & Evaluation with Scheduler Integration
########################################
def train_and_evaluate(
    model,
    train_loader,
    val_loader,
    test_loader,
    objective_metric="accuracy",
    objective_on="val",
    device=DEVICE,
    num_epochs=NUM_EPOCHS,
    early_stop_patience=float("nan"),
    scheduler_params=None
):
    """
    Trains the model for num_epochs and returns final metrics.
    Uses a scheduler if scheduler_params are provided. Supports early stopping based on the chosen objective.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3,betas=(0.85,0.999))  # Base LR will be overridden by scheduler

    # Determine scheduler type and build StepLRScheduler if requested.
    scheduler = None
    if scheduler_params is not None:
        sched_type = scheduler_params.get("scheduler_type", "step")
        if sched_type == "step":
            # Derive total_steps from training data: total_steps = num_epochs * len(train_loader)
            total_steps = scheduler_params.get("total_steps", num_epochs * len(train_loader))
            warmup_steps = scheduler_params.get("warmup_steps", 0)
            init_lr = scheduler_params.get("init_lr", 1e-4)
            peak_lr = scheduler_params.get("peak_lr", 1e-3)
            end_lr = scheduler_params.get("end_lr", 1e-5)
            warmup_type = scheduler_params.get("warmup_type", "cosine")
            decay_type = scheduler_params.get("decay_type", "cosine")
            # Build step-wise scheduler
            scheduler = StepLRScheduler(
                optimizer=optimizer,
                total_steps=total_steps,
                warmup_steps=warmup_steps,
                init_lr=init_lr,
                peak_lr=peak_lr,
                end_lr=end_lr,
                warmup_type=warmup_type,
                decay_type=decay_type
            )
        else:
            # For our purposes, we prefer step-wise; other scheduler types can be added similarly.
            scheduler = None

    best_score = -float("inf")
    best_state = None
    epochs_no_improve = 0
    measure_loader = val_loader if objective_on == "val" else test_loader

    for epoch in range(1, num_epochs + 1):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if scheduler is not None:
                scheduler.step()

        # End of epoch evaluation
        metrics_dict = compute_metrics(model, measure_loader, device)
        current_score = metrics_dict.get(objective_metric, 0.0)
        vprint(f"[Epoch {epoch}] {objective_on} {objective_metric} = {current_score:.4f}", level=2)

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
            if current_score > best_score:
                best_score = current_score
                best_state = copy.deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)

    val_metrics = compute_metrics(model, val_loader, device) if val_loader is not None else {}
    test_metrics = compute_metrics(model, test_loader, device) if test_loader is not None else {}

    return val_metrics, test_metrics, copy.deepcopy(model.state_dict())

########################################
# 4) Bayesian Optimization Objective Function
########################################
def objective_function(peak_lr, init_lr_frac, end_lr_frac, warmup_steps_frac, weight_decay):
    """
    Objective function for Bayesian Optimization.
    
    Hyperparameters to tune:
      - peak_lr: absolute peak learning rate.
      - init_lr_frac: fraction of peak_lr to use as the initial learning rate during warmup.
      - end_lr_frac: fraction of peak_lr to use as the final learning rate.
      - warmup_steps_frac: fraction of total training steps to use for warmup.
      - weight_decay: optimizer weight decay.

    Returns the validation metric (by default, accuracy).
    """
    # Derive actual learning rates:
    init_lr = init_lr_frac * peak_lr
    end_lr = end_lr_frac * peak_lr

    # Load datasets (paths can be adjusted as needed)
    train_path = r"kmnist\results_augmentation_search2\kmnist_train_augmented_tensorset.pth"
    val_path   = r"kmnist\saved_tensorsets\kmnist_val_tensorset.pth"
    test_path  = r"kmnist\saved_tensorsets\kmnist_test_tensorset.pth"
    train_tds, val_tds, test_tds = load_tensor_datasets(train_path, val_path, test_path)
    train_loader, val_loader, test_loader = get_dataloaders(train_tds, val_tds, test_tds, BATCH_SIZE)

    # Derive total steps and compute warmup_steps from the fraction.
    total_steps = NUM_EPOCHS * len(train_loader)
    warmup_steps = int(warmup_steps_frac * total_steps)

    # Build scheduler parameters dictionary for step-wise scheduling.
    scheduler_params = {
        "scheduler_type": "step",
        "total_steps": total_steps,
        "warmup_steps": warmup_steps,
        "init_lr": init_lr,
        "peak_lr": peak_lr,
        "end_lr": end_lr,
        "warmup_type": "cosine",    # You may also consider tuning this.
        "decay_type": "cosine"
    }

    # Set optimizer parameters
    optimizer_params = {
        "optimizer_class": optim.AdamW,
        "lr": peak_lr,  # initial LR is overridden by scheduler
        "weight_decay": weight_decay
    }

    # Build model and get its initial state.
    from model import EfficientCNNwAttn  # Ensure your model file is available.
    num_classes = int(train_tds.tensors[1].max().item() + 1)
    model = EfficientCNNwAttn(num_classes=num_classes).to(DEVICE)
    torch.manual_seed(42)
    init_state = copy.deepcopy(model.state_dict())

    # Train model using our modular training function with the step-based scheduler.
    val_metrics, test_metrics, _ = train_and_evaluate(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        objective_metric=OBJECTIVE_METRIC,
        objective_on="val",
        device=DEVICE,
        num_epochs=NUM_EPOCHS,
        early_stop_patience=2,
        scheduler_params=scheduler_params  # This uses the StepLRScheduler internally.
    )

    score = val_metrics.get(OBJECTIVE_METRIC, 0.0)
    vprint(f"[objective_function] Params: peak_lr={peak_lr:.6f}, init_lr={init_lr:.6f}, end_lr={end_lr:.6f}, warmup_steps={warmup_steps}, weight_decay={weight_decay:.6f} => {OBJECTIVE_METRIC} = {score:.4f}")
    return score

########################################
# 5) Run Bayesian Optimization
########################################
def run_bayes_opt():
    """
    Runs Bayesian Optimization using the objective_function above.
    Hyperparameter bounds (pbounds) are set such that:
      - peak_lr: absolute value between 1e-3 and 1e-2.
      - init_lr_frac: fraction of peak_lr between 0.1 and 0.9.
      - end_lr_frac: fraction of peak_lr between 0.0 and 0.5.
      - warmup_steps_frac: fraction between 0.0 and 0.5.
      - weight_decay: between 1e-6 and 1e-3.
    """
    pbounds = {
        "peak_lr": (1e-3, 1e-2),
        "init_lr_frac": (0.1, 0.9),
        "end_lr_frac": (0.0, 0.5),
        "warmup_steps_frac": (0.0, 0.5),
        "weight_decay": (1e-6, 1e-3)
    }
    bo = BayesianOptimization(f=objective_function, pbounds=pbounds, random_state=42)
    bo.maximize(init_points=5, n_iter=20)
    vprint(f"Best found: {bo.max['params']} => {bo.max['target']:.4f}")
    return bo

########################################
# 6) MAIN CLI / Callable Pipeline
########################################
def main():
    vprint("[main] Starting modular Bayesian Optimization pipeline...", level=1)
    # For demonstration, run Bayesian Optimization
    bo = run_bayes_opt()
    vprint("[main] Bayesian Optimization complete.", level=1)
    
    # Optionally, after BO you could train a final model with the best parameters.
    # (Here we only print out the best hyperparameters found.)
    best_params = bo.max["params"]
    vprint(f"[main] Best hyperparameters: {best_params}", level=1)


if __name__ == "__main__":
    main()
