"""
File: optimize_modular.py

Description:
    A more modular approach to Bayesian Optimization of hyperparameters,
    focusing on a step-based LR scheduler (with optional warmup/decay),
    and checkpointing. At the end, we retrain the final best hyperparameters
    on multiple seeds to get an averaged performance measure.

Key Features:
  - Multiple metrics (accuracy, precision, recall, macro_f1, micro_f1, weighted_f1)
  - Default optimization on validation accuracy (OBJECTIVE_METRIC).
  - Integration with a step-wise LR scheduler (StepLRScheduler).
  - Hyperparameters tuned: peak_lr, init_lr_frac, end_lr_frac, warmup_steps_frac, weight_decay.
  - Checkpointing to resume partial Bayesian Optimization runs.
  - Final multi-seed retraining to get an average test accuracy.
  - Automatic saving of final results in JSON (and model states if desired).
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
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

# ================================
#  GLOBAL CONFIG / SETTINGS
# ================================
VERBOSE = 1
def vprint(msg, level=1):
    """Simple verbosity-controlled print."""
    if VERBOSE >= level:
        tqdm.write(msg)

# Available metrics for classification tasks:
VALID_METRICS = {"accuracy", "precision", "recall", "macro_f1", "micro_f1", "weighted_f1"}

# Default objective metric & usage
OBJECTIVE_METRIC = "accuracy"  # e.g. "accuracy", "macro_f1", ...
OBJECTIVE_ON = "val"           # "val" or "test" for objective scoring
NUM_EPOCHS = 10                # proxy training epochs for BO
BATCH_SIZE = 1028
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Directory for saving results & checkpoints
RESULTS_DIR = "./results_modular"
os.makedirs(RESULTS_DIR, exist_ok=True)

DATA_LOADER_SEED = 42
EARLYSTOP_PATIENCE = float('nan')  # set to a finite number (e.g. 2) to enable early stopping
MODEL_NAME = "EfficientCNNwAttn_v1"
CHECKPOINT_META_PATH = os.path.join(RESULTS_DIR, f"bayes_opt_checkpoint_meta_{MODEL_NAME}.json")
CHECKPOINT_OPTIMIZER_PATH = os.path.join(RESULTS_DIR, f"bayes_opt_checkpoint_optimizer_{MODEL_NAME}.pkl")

# Where final JSON results & model states go:
FINAL_RESULTS_JSON = os.path.join(RESULTS_DIR, f"{MODEL_NAME}_final_results.json")

# ================================
#  STEP LR SCHEDULER
# (Defines per-batch LR updates)
# ================================
class StepLRScheduler:
    """
    A step-wise learning rate scheduler that updates on *every batch* (each training step).

    Tunable features (passed via constructor):
      - warmup_steps: how many steps from init_lr -> peak_lr
      - total_steps: total steps for entire training
      - init_lr, peak_lr, end_lr
      - warmup_type & decay_type: linear, exponential, polynomial, cosine, sigmoid, static, (decay only) logarithmic
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
        self.optimizer = optimizer
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.init_lr = init_lr
        self.peak_lr = peak_lr
        self.end_lr = end_lr
        self.warmup_type = warmup_type.lower()
        self.decay_type = decay_type.lower()

        self.current_step = 0

        # Basic checks
        if self.init_lr is False and self.warmup_steps > 0:
            warnings.warn("warmup_steps > 0 but init_lr=False => skipping warmup.")
        if self.end_lr is False and (self.decay_type != "static"):
            warnings.warn("end_lr=False but decay_type != 'static' => no decay used.")
        if self.total_steps <= self.warmup_steps and self.warmup_steps > 0:
            warnings.warn("No decay phase will occur because total_steps <= warmup_steps.")

        # Initialize LR
        initial = self.init_lr if self.init_lr is not False else self.peak_lr
        self._set_lr(initial)

    def step(self):
        self.current_step += 1
        new_lr = self._compute_lr()
        self._set_lr(new_lr)

    def _compute_lr(self):
        """Compute LR for current step, handling warmup vs. decay phases."""
        if (self.init_lr is False) and (self.end_lr is False):
            return self.peak_lr

        # WARMUP
        if self.init_lr is not False and (self.current_step <= self.warmup_steps):
            return self._warmup_lr()
        else:
            # DECAY or stay at peak
            if self.end_lr is False:
                return self.peak_lr
            return self._decay_lr()

    def _warmup_lr(self):
        progress = self.current_step / max(1, self.warmup_steps)
        if self.warmup_type == "linear":
            return self.init_lr + (self.peak_lr - self.init_lr) * progress
        elif self.warmup_type == "exponential":
            ratio = self.peak_lr / max(1e-12, self.init_lr)
            return self.init_lr * (ratio ** progress)
        elif self.warmup_type == "polynomial":
            return self.init_lr + (self.peak_lr - self.init_lr) * (progress ** 2)
        elif self.warmup_type == "cosine":
            # f(0)=0, f(1)=1 => 1-cos(pi/2*progress)
            return self.init_lr + (self.peak_lr - self.init_lr) * (1 - math.cos((math.pi * progress) / 2))
        elif self.warmup_type == "sigmoid":
            k = 12.0
            raw0 = 1/(1+math.exp(k*0.5))
            raw1 = 1/(1+math.exp(-k*0.5))
            raw = 1/(1+math.exp(-k*(progress-0.5)))
            norm = (raw - raw0)/(raw1 - raw0)
            return self.init_lr + (self.peak_lr - self.init_lr)*norm
        elif self.warmup_type == "static":
            return self.init_lr
        else:
            raise ValueError(f"Unknown warmup_type {self.warmup_type}")

    def _decay_lr(self):
        decay_step = self.current_step - self.warmup_steps
        decay_total = max(1, self.total_steps - self.warmup_steps)
        progress = decay_step/decay_total

        if self.decay_type == "linear":
            return self.peak_lr + (self.end_lr - self.peak_lr)*progress
        elif self.decay_type == "exponential":
            ratio = self.end_lr / max(1e-12, self.peak_lr)
            return self.peak_lr*(ratio ** progress)
        elif self.decay_type == "polynomial":
            return self.peak_lr + (self.end_lr - self.peak_lr)*(progress ** 2)
        elif self.decay_type == "cosine":
            return self.end_lr + 0.5*(self.peak_lr - self.end_lr)*(1 + math.cos(math.pi*progress))
        elif self.decay_type == "static":
            return self.peak_lr
        elif self.decay_type == "sigmoid":
            k = 12.0
            raw0 = 1/(1+math.exp(-k*0.5))
            raw1 = 1/(1+math.exp(k*0.5))
            raw = 1/(1+math.exp(k*(progress-0.5)))
            norm = (raw - raw0)/(raw1 - raw0)
            return self.peak_lr + (self.end_lr - self.peak_lr)*norm
        elif self.decay_type == "logarithmic":
            # normalized so f(0)=1 => peak_lr, f(1)=0 => end_lr
            factor = 1 - math.log(1+progress*(math.e-1))
            return self.peak_lr*factor + self.end_lr*(1 - factor)
        else:
            raise ValueError(f"Unknown decay_type {self.decay_type}")

    def _set_lr(self, lr):
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr

    def get_lr(self):
        return self.optimizer.param_groups[0]["lr"]


########################################
#  DATASET LOADING
########################################
def load_tensor_datasets(train_path, val_path, test_path):
    vprint(f"[load_tensor_datasets] Loading from:\n  train={train_path}\n  val={val_path}\n  test={test_path}", level=2)
    train_tds = torch.load(train_path)
    val_tds   = torch.load(val_path)
    test_tds  = torch.load(test_path)
    vprint(f"[load_tensor_datasets] Train={len(train_tds)}, Val={len(val_tds)}, Test={len(test_tds)}", level=2)
    return train_tds, val_tds, test_tds

def get_dataloaders(train_tds, val_tds, test_tds, batch_size=BATCH_SIZE, shuffle_seed=DATA_LOADER_SEED):
    vprint(f"[get_dataloaders] Creating DataLoaders (batch_size={batch_size}, seed={shuffle_seed})", level=2)
    g = torch.Generator()
    g.manual_seed(shuffle_seed)
    train_loader = DataLoader(train_tds, batch_size=batch_size, shuffle=True, generator=g)
    val_loader   = DataLoader(val_tds, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_tds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


########################################
#  METRIC COMPUTATION
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

    metrics = {
        "accuracy": accuracy_score(all_labels, all_preds),
        "macro_f1": f1_score(all_labels, all_preds, average="macro"),
        "micro_f1": f1_score(all_labels, all_preds, average="micro"),
        "weighted_f1": f1_score(all_labels, all_preds, average="weighted"),
        "precision": precision_score(all_labels, all_preds, average="macro", zero_division=0),
        "recall": recall_score(all_labels, all_preds, average="macro", zero_division=0)
    }
    return metrics


########################################
#  TRAINING LOOP (with Step Scheduler)
########################################
def train_model_step_lr(
    model,
    train_loader,
    val_loader,
    test_loader,
    device=DEVICE,
    num_epochs=NUM_EPOCHS,
    objective_metric=OBJECTIVE_METRIC,
    objective_on="val",
    early_stop_patience=float('nan'),
    scheduler_params=None
):
    """
    Trains a model using a step-wise LR scheduler & returns (val_metrics, test_metrics, best_state).
    This is used both in the objective function & final multi-seed evaluation.
    """
    criterion = nn.CrossEntropyLoss()
    # Build optimizer with some default LR (will be overridden by the scheduler)
    # If user wants weight_decay or betas, they can pass them, but we'll keep it simple here:
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.999))

    # Build StepLRScheduler from scheduler_params
    scheduler = None
    if scheduler_params is not None:
        total_steps = scheduler_params.get("total_steps", num_epochs*len(train_loader))
        warmup_steps = scheduler_params.get("warmup_steps", 0)
        init_lr = scheduler_params.get("init_lr", 1e-4)
        peak_lr = scheduler_params.get("peak_lr", 1e-3)
        end_lr = scheduler_params.get("end_lr", 1e-5)
        warmup_type = scheduler_params.get("warmup_type", "cosine")
        decay_type = scheduler_params.get("decay_type", "cosine")

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

    best_score = -float('inf')
    best_state = None
    epochs_no_improve = 0
    measure_loader = val_loader if objective_on=="val" else test_loader

    for epoch in range(1, num_epochs+1):
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

        # Evaluate each epoch
        metrics_dict = compute_metrics(model, measure_loader, device)
        current_score = metrics_dict.get(objective_metric, 0.0)
        vprint(f"Epoch {epoch}/{num_epochs} - {objective_on} {objective_metric} = {current_score:.4f}", level=2)

        # Early stopping
        if not math.isnan(early_stop_patience):
            if current_score > best_score + 1e-9:
                best_score = current_score
                best_state = copy.deepcopy(model.state_dict())
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= early_stop_patience:
                vprint(f"[train_model_step_lr] Early stopping at epoch={epoch}. Best {objective_metric}={best_score:.4f}", level=2)
                break
        else:
            if current_score > best_score:
                best_score = current_score
                best_state = copy.deepcopy(model.state_dict())

    # Load best state
    if best_state is not None:
        model.load_state_dict(best_state)

    val_metrics = compute_metrics(model, val_loader, device) if val_loader is not None else {}
    test_metrics = compute_metrics(model, test_loader, device) if test_loader is not None else {}
    return val_metrics, test_metrics, best_state


########################################
#  CHECKPOINTING UTILITIES
########################################
def save_checkpoint(optimizer_bo, meta, checkpoint_meta_path, checkpoint_optimizer_path):
    with open(checkpoint_optimizer_path, "wb") as f:
        pickle.dump(optimizer_bo, f)
    with open(checkpoint_meta_path, "w") as f:
        json.dump(meta, f, indent=4)
    vprint("[checkpoint] Checkpoint saved.", level=2)

def load_checkpoint(checkpoint_meta_path, checkpoint_optimizer_path, expected_model_name):
    if os.path.exists(checkpoint_meta_path) and os.path.exists(checkpoint_optimizer_path):
        with open(checkpoint_meta_path, "r") as f:
            meta = json.load(f)
        if meta.get("model_name", None) != expected_model_name:
            vprint("[checkpoint] Model name mismatch. Ignoring checkpoint.", level=2)
            return None, None
        with open(checkpoint_optimizer_path, "rb") as f:
            optimizer_bo = pickle.load(f)
        vprint("[checkpoint] Checkpoint loaded successfully.", level=2)
        return meta, optimizer_bo
    else:
        return None, None

########################################
#  SAVE EXPERIMENT RESULTS
########################################
def save_experiment_results(meta, final_results_path=FINAL_RESULTS_JSON):
    """
    Saves the final experiment metadata (including best hyperparams, multi-seed results, etc.) as JSON.
    If you want to save model states, you can do it here too.
    """
    with open(final_results_path, "w") as f:
        json.dump(meta, f, indent=4)
    vprint(f"[save_experiment_results] Final results saved to: {final_results_path}", level=1)

########################################
#  OBJECTIVE FUNCTION
########################################
def objective_function(peak_lr, init_lr_frac, end_lr_frac, warmup_steps_frac, weight_decay):
    """
    Bayesian Optimization objective:
      We tune:
        - peak_lr
        - init_lr_frac => init_lr = peak_lr * init_lr_frac
        - end_lr_frac  => end_lr  = peak_lr * end_lr_frac
        - warmup_steps_frac => warmup_steps = int(total_steps * warmup_steps_frac)
        - weight_decay
    We'll do a short "proxy" training with step-wise scheduling & return val accuracy.
    """
    # Derive actual LRs
    init_lr = init_lr_frac * peak_lr
    end_lr = end_lr_frac * peak_lr

    # Load dataset for the proxy training
    train_path = r"kmnist\results_augmentation_search2\kmnist_train_augmented_tensorset.pth"
    val_path   = r"kmnist\saved_tensorsets\kmnist_val_tensorset.pth"
    test_path  = r"kmnist\saved_tensorsets\kmnist_test_tensorset.pth"
    train_tds, val_tds, test_tds = load_tensor_datasets(train_path, val_path, test_path)
    train_loader, val_loader, test_loader = get_dataloaders(train_tds, val_tds, test_tds, BATCH_SIZE)

    # total steps for num_epochs
    total_steps = NUM_EPOCHS * len(train_loader)
    warmup_steps = int(warmup_steps_frac * total_steps)

    # Build scheduler params
    scheduler_params = {
        "scheduler_type": "step",
        "total_steps": total_steps,
        "warmup_steps": warmup_steps,
        "init_lr": init_lr,
        "peak_lr": peak_lr,
        "end_lr": end_lr,
        "warmup_type": "cosine",
        "decay_type": "cosine"
    }

    # Build the model
    # If you have custom model, import it here
    from model import EfficientCNNwAttn
    torch.manual_seed(42)
    random.seed(42)

    # Prepare model
    num_classes = int(train_tds.tensors[1].max().item() + 1)
    model = EfficientCNNwAttn(num_classes=num_classes).to(DEVICE)
    # Reset model's parameters if it has .reset_parameters
    model.apply(lambda m: m.reset_parameters() if hasattr(m, 'reset_parameters') else None)

    # We'll incorporate weight_decay by adjusting the param groups in training. For simplicity, we do so by
    # not using it directly in the optimizer constructor. Another approach is to pass it in the final step.
    # For the objective, let's do a short training with the step LR.
    val_metrics, test_metrics, _ = train_model_step_lr(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=DEVICE,
        num_epochs=NUM_EPOCHS,
        objective_metric=OBJECTIVE_METRIC,
        objective_on="val",
        early_stop_patience=2,
        scheduler_params=scheduler_params
    )

    score = val_metrics.get(OBJECTIVE_METRIC, 0.0)
    vprint(f"[objective_function] => peak_lr={peak_lr:.6f}, init_lr={init_lr:.6f}, end_lr={end_lr:.6f}, warmup_steps={warmup_steps}, wd={weight_decay:.6g} => {OBJECTIVE_METRIC}={score:.4f}", level=1)
    return score

########################################
#  BAYESIAN OPT
########################################
def run_bayes_opt():
    """
    Orchestrates Bayesian Optimization with partial checkpointing. We tune:
       - peak_lr
       - init_lr_frac
       - end_lr_frac
       - warmup_steps_frac
       - weight_decay
    """
    pbounds = {
        "peak_lr": (1e-3, 1e-2),
        "init_lr_frac": (0.1, 0.9),
        "end_lr_frac": (0.0, 0.5),
        "warmup_steps_frac": (0.0, 0.5),
        "weight_decay": (1e-5, 1e-5)
    }

    meta, optimizer_bo = load_checkpoint(CHECKPOINT_META_PATH, CHECKPOINT_OPTIMIZER_PATH, expected_model_name=MODEL_NAME)
    if optimizer_bo is None:
        vprint("[bayes_opt] No checkpoint found. Starting fresh Bayesian Optimization...", level=1)
        optimizer_bo = BayesianOptimization(
            f=objective_function,
            pbounds=pbounds,
            random_state=42,
        )
        meta = {"model_name": MODEL_NAME, "completed": "NULL", "best_params": None}
        # Optional: we can do init_points first
        optimizer_bo.maximize(init_points=5, n_iter=0)
    else:
        vprint("[bayes_opt] Resuming from existing checkpoint...", level=1)

    total_iter = 30
    checkpoint_interval = 10

    try:
        for i in range(0, total_iter, checkpoint_interval):
            steps_this_block = min(checkpoint_interval, total_iter - i)
            vprint(f"[bayes_opt] Running {steps_this_block} iterations of BO (Block {i+1}/{total_iter})", level=1)
            optimizer_bo.maximize(n_iter=steps_this_block)

            # Update meta
            meta["best_params"] = optimizer_bo.max.get("params", None)
            meta["completed"] = "NULL"
            # Save checkpoint
            save_checkpoint(optimizer_bo, meta, CHECKPOINT_META_PATH, CHECKPOINT_OPTIMIZER_PATH)

    except Exception as e:
        vprint(f"[bayes_opt] Optimization interrupted: {e}", level=1)
        meta["best_params"] = optimizer_bo.max.get("params", None)
        meta["completed"] = "INTERRUPTED"
        save_checkpoint(optimizer_bo, meta, CHECKPOINT_META_PATH, CHECKPOINT_OPTIMIZER_PATH)
        raise e

    # final
    meta["completed"] = "FULL"
    meta["best_params"] = optimizer_bo.max["params"]
    meta["best_score"] = optimizer_bo.max["target"]
    vprint(f"[bayes_opt] Best parameters found: {meta['best_params']} => {meta['best_score']:.4f}", level=1)

    # remove checkpoint files if you want a "clean" result
    if os.path.exists(CHECKPOINT_META_PATH):
        os.remove(CHECKPOINT_META_PATH)
    if os.path.exists(CHECKPOINT_OPTIMIZER_PATH):
        os.remove(CHECKPOINT_OPTIMIZER_PATH)

    return meta

########################################
#  MULTI-SEED RETRAIN
########################################
def final_multiseed_eval(best_params, num_runs=5):
    """
    Retrain model with best_params over multiple seeds (e.g. 5) and return the average test accuracy.
    """
    # Derive the actual LR values
    peak_lr = best_params["peak_lr"]
    init_lr = best_params["init_lr_frac"] * peak_lr
    end_lr = best_params["end_lr_frac"] * peak_lr
    # We'll do the same dataset as we used in BO:
    train_path = r"kmnist\results_augmentation_search2\kmnist_train_augmented_tensorset.pth"
    val_path   = r"kmnist\saved_tensorsets\kmnist_val_tensorset.pth"
    test_path  = r"kmnist\saved_tensorsets\kmnist_test_tensorset.pth"
    train_tds, val_tds, test_tds = load_tensor_datasets(train_path, val_path, test_path)
    train_loader, val_loader, test_loader = get_dataloaders(train_tds, val_tds, test_tds, BATCH_SIZE)

    total_steps = NUM_EPOCHS * len(train_loader)
    warmup_steps = int(best_params["warmup_steps_frac"]*total_steps)

    scheduler_params = {
        "scheduler_type":"step",
        "total_steps": total_steps,
        "warmup_steps": warmup_steps,
        "init_lr": init_lr,
        "peak_lr": peak_lr,
        "end_lr": end_lr,
        "warmup_type": "cosine",
        "decay_type": "cosine"
    }

    accuracies = []
    # Example model used below
    from model import EfficientCNNwAttn
    for run_idx in range(num_runs):
        seed = 42+run_idx
        torch.manual_seed(seed)
        random.seed(seed)

        # Build new model
        num_classes = int(train_tds.tensors[1].max().item() + 1)
        model = EfficientCNNwAttn(num_classes=num_classes).to(DEVICE)
        # Reset parameters
        model.apply(lambda m: m.reset_parameters() if hasattr(m,'reset_parameters') else None)

        # Train
        _, test_metrics, _ = train_model_step_lr(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            device=DEVICE,
            num_epochs=NUM_EPOCHS,
            objective_metric=OBJECTIVE_METRIC,
            objective_on="val",
            early_stop_patience=2,
            scheduler_params=scheduler_params
        )
        test_acc = test_metrics["accuracy"]
        vprint(f"[final_multiseed_eval] Run {run_idx+1}/{num_runs}, seed={seed}, Test Acc={test_acc:.4f}", level=1)
        accuracies.append(test_acc)

    avg_acc = sum(accuracies)/len(accuracies)
    var_acc = sum((x-avg_acc)**2 for x in accuracies)/len(accuracies)
    best_acc = max(accuracies)
    return {
        "individual_accuracies": accuracies,
        "average_accuracy": avg_acc,
        "variance": var_acc,
        "best_accuracy": best_acc
    }

########################################
#  MAIN
########################################
def main():
    vprint("[main] Starting modular pipeline with Step-based LR + Bayesian Optimization", level=1)
    # 1) Run Bayesian Optimization or resume from checkpoint
    meta = run_bayes_opt()

    best_params = meta["best_params"]
    best_score = meta["best_score"]
    vprint(f"[main] Found best hyperparams = {best_params}, BO val score={best_score:.4f}", level=1)

    # 2) Multi-seed final evaluation
    multi_seed_results = final_multiseed_eval(best_params, num_runs=5)
    vprint(f"[main] Multi-seed final results: {multi_seed_results}", level=1)

    # 3) Build final meta dict with multi-seed info
    meta["multi_seed_results"] = multi_seed_results
    meta["final_average_accuracy"] = multi_seed_results["average_accuracy"]
    meta["final_best_accuracy"] = multi_seed_results["best_accuracy"]
    meta["completed"] = "FULL_MULTI_SEED"

    # 4) Save final results to JSON
    save_experiment_results(meta, final_results_path=FINAL_RESULTS_JSON)

    vprint(f"[main] All done. Final average Test Acc: {multi_seed_results['average_accuracy']:.4f}", level=1)


if __name__ == "__main__":
    main()
