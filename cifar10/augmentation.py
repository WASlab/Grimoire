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
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms as T
from tqdm import tqdm
from bayes_opt import BayesianOptimization  # pip install bayesian-optimization
import matplotlib.pyplot as plt

from model import EfficientCNN

# =============== Global Settings & Hyperparameters ===============
NUM_EPOCHS_PROXY      = 5          # proxy training epochs for objective evaluation
NUM_EPOCHS_FINAL      = 10         # not used here directly, final training uses 5 epochs
FINAL_TRAIN_EPOCHS    = 5          # final training epochs for evaluation on test set
BATCH_SIZE            = 1028
DEVICE                = "cuda" if torch.cuda.is_available() else "cpu"
SAVED_TENSORSETS_DIR  = "./CIFAR10/saved_tensorsets"
RESULTS_DIR           = "./CIFAR10/results_augmentation_search"
os.makedirs(RESULTS_DIR, exist_ok=True)
DATA_LOADER_SEED      = 12345      # fixed seed for data loading reproducibility

# ---------------------------
# Helper Transform Wrapper: Applies a transform with probability p (as nn.Module)
# ---------------------------
class RandomApplyCustom(nn.Module):
    def __init__(self, transform, p=0.5):
        super(RandomApplyCustom, self).__init__()
        self.transform = transform
        self.p = p
        
    def forward(self, img):
        if torch.rand(1, device=img.device).item() < self.p:
            return self.transform(img)
        return img

# ---------------------------
# Custom Transform: Gaussian Noise (as nn.Module)
# ---------------------------
class GaussianNoise(nn.Module):
    def __init__(self, std=0.05):
        super(GaussianNoise, self).__init__()
        self.std = std
        
    def forward(self, tensor):
        noise = torch.randn_like(tensor, device=tensor.device) * self.std
        return tensor + noise

# ---------------------------
# Batched MixUp: Vectorized mixup across a batch
# ---------------------------
def batched_mixup(batch, mixup_prob, mixup_alpha):
    N = batch.size(0)
    rand = torch.rand(N, device=batch.device)
    mixup_mask = rand < mixup_prob
    if mixup_mask.sum() > 0:
        beta = torch.distributions.Beta(mixup_alpha, mixup_alpha)
        lam = beta.sample([N]).to(batch.device)
        perm = torch.randperm(N, device=batch.device)
        batch[mixup_mask] = (
            lam[mixup_mask].view(-1, 1, 1, 1) * batch[mixup_mask] +
            (1 - lam[mixup_mask].view(-1, 1, 1, 1)) * batch[perm][mixup_mask]
        )
    return batch

# ---------------------------
# Batched CutOut: Vectorized cutout applied to a batch of images.
#  [# CHANGED] to allow fill_value != 0
# ---------------------------
def batched_cutout(batch, cutout_frac, cutout_prob, fill_value=0.0):
    B, C, H, W = batch.shape
    mask = torch.ones((B, 1, H, W), device=batch.device, dtype=batch.dtype)
    apply_mask = torch.rand(B, device=batch.device) < cutout_prob
    indices = torch.nonzero(apply_mask).squeeze(1)
    if indices.numel() > 0:
        cutout_size = int(cutout_frac * H)
        if cutout_size < 1:
            return batch
        for idx in indices.tolist():
            t = random.randint(0, H - cutout_size)
            l = random.randint(0, W - cutout_size)
            mask[idx, :, t:t+cutout_size, l:l+cutout_size] = 0
    # If fill_value = 0.0, this is standard “mask out” approach
    # If fill_value != 0.0, it fills the cut area with that constant
    return batch * mask + fill_value * (1 - mask)

# ---------------------------
# Data Loading Functions (assumes tensor datasets already exist)
# ---------------------------
def load_tensor_datasets(save_dir=SAVED_TENSORSETS_DIR):
    tqdm.write(f"[load_tensor_datasets] Loading TensorDatasets from '{save_dir}'...")
    train_tds = torch.load(os.path.join(save_dir, "CIFAR10_train_tensorset.pth"))
    val_tds   = torch.load(os.path.join(save_dir, "CIFAR10_val_tensorset.pth"))
    test_tds  = torch.load(os.path.join(save_dir, "CIFAR10_test_tensorset.pth"))
    tqdm.write(f"[load_tensor_datasets] Loaded: Train={len(train_tds)}, Val={len(val_tds)}, Test={len(test_tds)}")
    return train_tds, val_tds, test_tds

def get_dataloaders(train_tds, val_tds, test_tds, batch_size=BATCH_SIZE, shuffle_seed=DATA_LOADER_SEED):
    tqdm.write(f"[get_dataloaders] Creating DataLoaders with batch_size={batch_size}...")
    g = torch.Generator()
    g.manual_seed(shuffle_seed)
    train_loader = DataLoader(train_tds, batch_size=batch_size, shuffle=True, generator=g)
    val_loader   = DataLoader(val_tds, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_tds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader

# ---------------------------
# Augmentation Pipeline Builder (GPU-accelerated)
#   [# CHANGED] to handle color jitter, scale, cutout_fill, etc.
# ---------------------------
def build_augmentation_pipeline(policy):
    """
    Build a GPU-accelerated augmentation pipeline as nn.Sequential from a given policy dictionary.
    Expected keys (besides mixup and cutout handled separately):
      rotate_prob, rotate_deg, translate_prob, translate_frac, flip_prob,
      noise_prob, noise_std, perspective_prob, perspective_distortion,
      color_jitter_prob, color_jitter_strength,
      scale_prob, cutout_fill, ...
    """
    ops = []
    
    # Rotation
    if policy.get("rotate_deg", 0) > 0:
        rotate_deg = policy["rotate_deg"]
        rotate_transform = T.RandomRotation(degrees=(-rotate_deg, rotate_deg))
        ops.append(RandomApplyCustom(rotate_transform, p=policy["rotate_prob"]))

    # Translation
    if policy.get("translate_frac", 0) > 0:
        translate_transform = T.RandomAffine(
            degrees=0,
            translate=(policy["translate_frac"], policy["translate_frac"])
        )
        ops.append(RandomApplyCustom(translate_transform, p=policy["translate_prob"]))

    # Horizontal Flip
    ops.append(RandomApplyCustom(T.RandomHorizontalFlip(p=1.0), p=policy["flip_prob"]))

    # Perspective
    if policy.get("perspective_distortion", 0) > 0:
        persp_transform = T.RandomPerspective(distortion_scale=policy["perspective_distortion"], p=1.0)
        ops.append(RandomApplyCustom(persp_transform, p=policy["perspective_prob"]))

    # ColorJitter
    if policy.get("color_jitter_strength", 0) > 0:
        cjs = policy["color_jitter_strength"]
        color_jitter = T.ColorJitter(brightness=cjs, contrast=cjs, saturation=cjs, hue=0.05)
        ops.append(RandomApplyCustom(color_jitter, p=policy["color_jitter_prob"]))

    # Random Resized Crop for scale
    # We'll fix a scale range of [0.8, 1.2].
    # If your images are 32x32, be sure T.RandomResizedCrop expects them to remain 32x32.
    if policy.get("scale_prob", 0) > 0:
        scale_transform = T.RandomResizedCrop(
            (32, 32),  # keep final size at 32x32
            scale=(0.8, 1.2),
            ratio=(0.9, 1.1)  # keep aspect ratio near 1
        )
        ops.append(RandomApplyCustom(scale_transform, p=policy["scale_prob"]))

    # Gaussian Noise
    if policy.get("noise_std", 0) > 0:
        ops.append(RandomApplyCustom(GaussianNoise(std=policy["noise_std"]), p=policy["noise_prob"]))

    augmentation_pipeline = nn.Sequential(*ops)
    return augmentation_pipeline

# ---------------------------
# Batched Dataset Augmentation Function (Batched Processing)
# ---------------------------
def create_augmented_dataset_batched(original_dataset, augmentation_pipeline, num_copies, policy, batch_size_augment=2056):
    """
    Collect all images from original_dataset and apply augmentation in batches.
    Incorporates mixup and cutout in a vectorized manner.
    """
    all_images = []
    all_labels = []
    for i in range(len(original_dataset)):
        img, label = original_dataset[i]
        all_images.append(img.unsqueeze(0))
        all_labels.append(torch.tensor([label]))
    images = torch.cat(all_images, dim=0)  # (N, C, H, W)
    labels = torch.cat(all_labels, dim=0)  # (N,)
    
    augmented_batches = [images]  # include original images
    N = images.shape[0]
    
    mixup_prob = policy.get("mixup_prob", 0.0)
    mixup_alpha = policy.get("mixup_alpha", 0.2)
    cutout_prob = policy.get("cutout_prob", 0.0)
    cutout_size = policy.get("cutout_size", 0.0)
    cutout_fill = policy.get("cutout_fill", 0.0)
    
    for _ in range(num_copies):
        aug_images_list = []
        for start in range(0, N, batch_size_augment):
            end = start + batch_size_augment
            batch = images[start:end].to(DEVICE)
            # Apply pipeline
            batch = augmentation_pipeline(batch)
            # MixUp
            batch = batched_mixup(batch, mixup_prob, mixup_alpha)
            # CutOut
            batch = batched_cutout(batch, cutout_size, cutout_prob, fill_value=cutout_fill)
            aug_images_list.append(batch.cpu())
        aug_batch = torch.cat(aug_images_list, dim=0)
        augmented_batches.append(aug_batch)
        
    final_images = torch.cat(augmented_batches, dim=0)
    final_labels = labels.repeat(num_copies + 1)
    return TensorDataset(final_images, final_labels)

# ---------------------------
# Proxy Evaluation Function for Augmentation Policy
#   [# CHANGED] - we now fix a seed for each evaluation
# ---------------------------
def evaluate_augmentation_policy(policy_params, init_state, train_tds, val_tds, test_tds,
                                batch_size=BATCH_SIZE, num_epochs=NUM_EPOCHS_PROXY, seed=9999):
    """
    Train a fresh model from init_state on a newly augmented dataset, then compute val accuracy.
    We fix 'seed' inside this function to reduce noise across evaluations.
    """
    # [# CHANGED] Fix seed so that random augmentations are reproducible each time.
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    policy = {
        "rotate_prob": policy_params.get("rotate_prob", 0.0),
        "rotate_deg": abs(policy_params.get("rotate_deg", 0.0)),
        "translate_prob": policy_params.get("translate_prob", 0.0),
        "translate_frac": policy_params.get("translate_frac", 0.0),
        "flip_prob": policy_params.get("flip_prob", 0.0),
        "noise_prob": policy_params.get("noise_prob", 0.0),
        "noise_std": policy_params.get("noise_std", 0.0),
        "perspective_prob": policy_params.get("perspective_prob", 0.0),
        "perspective_distortion": policy_params.get("perspective_distortion", 0.0),
        "mixup_prob": policy_params.get("mixup_prob", 0.0),
        "mixup_alpha": policy_params.get("mixup_alpha", 0.2),
        "cutout_prob": policy_params.get("cutout_prob", 0.0),
        "cutout_size": policy_params.get("cutout_size", 0.0),
        "cutout_fill": policy_params.get("cutout_fill", 0.0),
        "color_jitter_prob": policy_params.get("color_jitter_prob", 0.0),
        "color_jitter_strength": policy_params.get("color_jitter_strength", 0.0),
        "scale_prob": policy_params.get("scale_prob", 0.0),
    }
    num_copies = max(1, int(round(policy_params.get("num_augmented_copies", 1))))

    # Build pipeline
    augmentation_pipeline = build_augmentation_pipeline(policy).to(DEVICE)
    
    # Create augmented training set
    augmented_train_tds = create_augmented_dataset_batched(
        train_tds, augmentation_pipeline, num_copies, policy
    )
    
    # DataLoaders
    train_loader, val_loader, _ = get_dataloaders(augmented_train_tds, val_tds, test_tds, batch_size)

    # Re-initialize a model with the same initial weights
    num_classes = int(train_tds.tensors[1].max().item() + 1)
    model = EfficientCNN(num_classes=num_classes).to(DEVICE)
    model.load_state_dict(init_state)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    
    # Train for num_epochs
    for epoch in range(num_epochs):
        model.train()
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    
    # Evaluate on validation set
    model.eval()
    total_correct, total_samples = 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)
    accuracy = total_correct / total_samples
    return accuracy

# ---------------------------
# Objective Function for Bayesian Optimization
#   [# CHANGED] - pass fixed seed to reduce noise
# ---------------------------
def objective_function(**kwargs):
    # Make sure we keep num_augmented_copies >= 1
    kwargs["num_augmented_copies"] = max(1, int(round(kwargs["num_augmented_copies"])))
    acc = evaluate_augmentation_policy(
        kwargs,
        initial_state,
        train_tds,
        val_tds,
        test_tds,
        batch_size=BATCH_SIZE,
        num_epochs=NUM_EPOCHS_PROXY,
        seed=9999  # [# CHANGED] for consistent evaluation
    )
    tqdm.write(f"[objective_function] Candidate params: {kwargs} => Val Accuracy: {acc:.4f}")
    return acc

# ---------------------------
# Function to force near-zero values to 0 for selected parameters
# ---------------------------
def zero_check(policy_params, threshold=0.01):
    """
    Force parameters that are below the threshold to 0.
    Exclude 'num_augmented_copies' and 'mixup_alpha' from this check by default,
    unless you'd like to include them.
    """
    modified_policy = policy_params.copy()
    for key in policy_params:
        if key not in ["num_augmented_copies", "mixup_alpha"]:
            val = policy_params[key]
            if isinstance(val, (float, int)) and (abs(val) < threshold):
                modified_policy[key] = 0.0
    return modified_policy

# ---------------------------
# [# ADDED] Function for sequential param-zero check
#   We take the best policy, then for each parameter in turn (except a few),
#   we set that parameter to zero if it improves validation accuracy.
# ---------------------------
def sequential_zero_check(initial_best_params, init_state,
                          train_tds, val_tds, test_tds,
                          param_list, best_val_acc, threshold=0.01,
                          batch_size=BATCH_SIZE, num_epochs=NUM_EPOCHS_PROXY,
                          seed=9999):
    """
    param_list: which params to consider zeroing (e.g. from the keys of pbounds).
    We do it in a naive sequential manner: for each param in param_list,
      - set param to 0
      - re-evaluate
      - if better, keep it at 0, else revert
    """
    current_params = initial_best_params.copy()
    
    for key in param_list:
        # skip if it's 'num_augmented_copies' or if the param is already zero
        if key in ["num_augmented_copies", "mixup_alpha"]:
            continue
        if abs(current_params.get(key, 0.0)) < threshold:
            # already near zero
            continue
        
        test_params = current_params.copy()
        test_params[key] = 0.0
        
        new_val_acc = evaluate_augmentation_policy(
            test_params, init_state, train_tds, val_tds, test_tds,
            batch_size=batch_size, num_epochs=num_epochs, seed=seed
        )
        if new_val_acc > best_val_acc:
            tqdm.write(f"[sequential_zero_check] Setting '{key}'=0 improved Val Acc {best_val_acc:.4f} -> {new_val_acc:.4f}")
            best_val_acc = new_val_acc
            current_params = test_params
        else:
            tqdm.write(f"[sequential_zero_check] Setting '{key}'=0 did NOT improve Val Acc ({new_val_acc:.4f} <= {best_val_acc:.4f}), reverting.")
    
    return current_params, best_val_acc


if __name__ == "__main__":
    # ---------------------------
    # Load datasets and set up model
    # ---------------------------
    train_tds, val_tds, test_tds = load_tensor_datasets(SAVED_TENSORSETS_DIR)
    num_classes = int(train_tds.tensors[1].max().item() + 1)
    
    # We fix the initial state for model so each evaluation starts identically
    torch.manual_seed(42)
    init_model = EfficientCNN(num_classes=num_classes)
    initial_state = copy.deepcopy(init_model.state_dict())
    
    # ---------------------------
    # Define the Bayesian Optimization search space
    # ---------------------------
    pbounds = {
        # --- Spatial Transformations ---
        "rotate_prob": (0.3, 0.9),
        "rotate_deg": (0, 25),  # typically we store angle as positive & apply +/- in builder
        "translate_prob": (0.5, 1.0),
        "translate_frac": (0.0, 0.25),
        "flip_prob": (0.5, 0.9),
        "perspective_prob": (0.3, 0.8),
        "perspective_distortion": (0.0, 0.5),

        # --- Color & Noise ---
        "color_jitter_prob": (0.5, 0.9),
        "color_jitter_strength": (0.1, 0.5),
        "noise_prob": (0.2, 0.6),
        "noise_std": (0.0, 0.1),

        # --- Advanced Augmentations ---
        "mixup_prob": (0.3, 0.7),
        "mixup_alpha": (0.2, 0.8),
        "cutout_prob": (0.5, 0.9),
        "cutout_size": (0.1, 0.4),
        "cutout_fill": (0.0, 0.3),   # fill range: black up to a gray level

        # --- Multi-Scale ---
        "scale_prob": (0.5, 0.9),

        # --- Number of augmented copies ---
        "num_augmented_copies": (8, 12),
    }
    
    # ---------------------------
    # Bayesian Optimization
    # ---------------------------
    optimizer = BayesianOptimization(
        f=objective_function,
        pbounds=pbounds,
        random_state=42
    )
    
    tqdm.write("[bayes_opt] Starting augmentation policy search with Bayesian Optimization...")
    optimizer.maximize(init_points=5, n_iter=50)
    
    best_params = optimizer.max["params"]
    best_params["num_augmented_copies"] = int(round(best_params["num_augmented_copies"]))
    best_val_acc = optimizer.max["target"]
    tqdm.write(f"[bayes_opt] Best augmentation parameters found:\n  {best_params} \n  => Val Accuracy: {best_val_acc:.4f}")
    
    # ---------------------------
    # Zero-Check: Force near-zero params to 0, re-test
    # ---------------------------
    zero_params = zero_check(best_params, threshold=0.01)
    zero_val_acc = evaluate_augmentation_policy(
        zero_params, initial_state, train_tds, val_tds, test_tds,
        batch_size=BATCH_SIZE, num_epochs=NUM_EPOCHS_PROXY, seed=9999
    )
    if zero_val_acc > best_val_acc:
        tqdm.write(f"[zero_check] Forcing near-zero values => improved Val Acc: {best_val_acc:.4f} -> {zero_val_acc:.4f}")
        best_params = zero_params
        best_val_acc = zero_val_acc
    else:
        tqdm.write(f"[zero_check] Forcing near-zero values did NOT improve (Val Acc: {zero_val_acc:.4f} <= {best_val_acc:.4f}). Keeping original.")

    # ---------------------------
    # [# ADDED] Sequential param-zero check
    # ---------------------------
    param_order = list(pbounds.keys())  # naive approach: step through all
    updated_params, updated_val_acc = sequential_zero_check(
        best_params,
        initial_state,
        train_tds, val_tds, test_tds,
        param_list=param_order,
        best_val_acc=best_val_acc,
        threshold=0.01,
        batch_size=BATCH_SIZE,
        num_epochs=NUM_EPOCHS_PROXY,
        seed=9999
    )
    if updated_val_acc > best_val_acc:
        best_params = updated_params
        best_val_acc = updated_val_acc
    
    # ---------------------------
    # [# ADDED] Check if using max num_augmented_copies helps
    # ---------------------------
    max_copies = int(pbounds["num_augmented_copies"][1])
    if max_copies != best_params["num_augmented_copies"]:
        temp_params = best_params.copy()
        temp_params["num_augmented_copies"] = max_copies
        
        max_copy_val_acc = evaluate_augmentation_policy(
            temp_params, initial_state, train_tds, val_tds, test_tds,
            batch_size=BATCH_SIZE, num_epochs=NUM_EPOCHS_PROXY, seed=9999
        )
        if max_copy_val_acc > best_val_acc:
            tqdm.write(f"[check_max_copies] Using max num_augmented_copies={max_copies} improved Val Acc {best_val_acc:.4f} -> {max_copy_val_acc:.4f}")
            best_params = temp_params
            best_val_acc = max_copy_val_acc
        else:
            tqdm.write(f"[check_max_copies] Using max copies did NOT improve (Val Acc: {max_copy_val_acc:.4f} <= {best_val_acc:.4f}). Keeping original.")

    tqdm.write(f"[final] Final best params after sequential zero check & max-copy check:\n  {best_params}")
    tqdm.write(f"[final] Final best Val Accuracy: {best_val_acc:.4f}")

    # ---------------------------
    # Create final augmented dataset using these best parameters
    # ---------------------------
    best_augmentation_pipeline = build_augmentation_pipeline(best_params).to(DEVICE)
    num_copies = best_params["num_augmented_copies"]
    tqdm.write("[final] Creating final augmented training dataset with the best augmentation policy...")
    
    final_augmented_train_tds = create_augmented_dataset_batched(
        train_tds,
        best_augmentation_pipeline,
        num_copies,
        best_params
    )
    
    # Save the final augmented dataset and policy parameters.
    final_save_path = os.path.join(RESULTS_DIR, "CIFAR10_train_augmented_tensorset.pth")
    torch.save(final_augmented_train_tds, final_save_path)
    tqdm.write(f"[final] Final augmented TensorDataset saved to: {final_save_path}")
    
    policy_save_path = os.path.join(RESULTS_DIR, "best_augmentation_policy.json")
    with open(policy_save_path, "w") as f:
        json.dump(best_params, f, indent=4)
    tqdm.write(f"[final] Best augmentation policy saved to: {policy_save_path}")
    
    # ======= FINAL MODEL TRAINING AND TEST EVALUATION =======
    final_train_loader, _, test_loader = get_dataloaders(
        final_augmented_train_tds, val_tds, test_tds, batch_size=BATCH_SIZE
    )
    
    # Initialize final model with the same initial state
    final_model = EfficientCNN(num_classes=num_classes).to(DEVICE)
    final_model.load_state_dict(initial_state)
    criterion = nn.CrossEntropyLoss()
    optimizer_final = optim.AdamW(final_model.parameters(), lr=1e-3)
    
    tqdm.write(f"[final] Training final model on augmented training dataset for {FINAL_TRAIN_EPOCHS} epochs...")
    for epoch in range(FINAL_TRAIN_EPOCHS):
        final_model.train()
        for images, labels in tqdm(final_train_loader, desc=f"Final Epoch {epoch+1}/{FINAL_TRAIN_EPOCHS}", leave=False):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer_final.zero_grad()
            outputs = final_model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer_final.step()
    
    # Evaluate final model on test set
    final_model.eval()
    total_correct, total_samples = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = final_model(images)
            preds = outputs.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)
    test_accuracy = total_correct / total_samples
    tqdm.write(f"[final] Final Test Accuracy: {test_accuracy:.4f}")
    
    tqdm.write("[main] Augmentation search and final evaluation complete. Enjoy your optimal augmented dataset and model!")
