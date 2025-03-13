"""
Module: augment_with_bayes.py
Location: grimoire/vision/augmentations/
Description: This module provides a fully modular pipeline for augmentation policy search
using Bayesian Optimization. It builds augmentation pipelines (with support for both RGB and grayscale),
creates augmented datasets in a batched, GPU-accelerated manner, evaluates policies via proxy training,
and finally trains a model on the best augmented dataset. It also includes utilities for “zero-checking”
parameters sequentially.
"""

import os
import copy
import json
import random
import math

import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms as T
from tqdm import tqdm
from bayes_opt import BayesianOptimization

# ---------------------------
# Helper Transform Classes
# ---------------------------
class RandomApplyCustom(nn.Module):
    """
    Wraps a transform so that it is applied with probability p.
    """
    def __init__(self, transform, p=0.5):
        super(RandomApplyCustom, self).__init__()
        self.transform = transform
        self.p = p

    def forward(self, img):
        if torch.rand(1, device=img.device).item() < self.p:
            return self.transform(img)
        return img

class GaussianNoise(nn.Module):
    """
    Adds Gaussian noise to the input tensor.
    """
    def __init__(self, std=0.05):
        super(GaussianNoise, self).__init__()
        self.std = std

    def forward(self, tensor):
        noise = torch.randn_like(tensor, device=tensor.device) * self.std
        return tensor + noise

# ---------------------------
# Batched Augmentation Functions
# ---------------------------
def batched_mixup(batch, mixup_prob, mixup_alpha):
    """
    Applies vectorized mixup across a batch.
    
    Args:
        batch (Tensor): Batch of images.
        mixup_prob (float): Probability of applying mixup per sample.
        mixup_alpha (float): Alpha parameter for Beta distribution.
    
    Returns:
        Tensor: Batch after mixup.
    """
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

def batched_cutout(batch, cutout_frac, cutout_prob, fill_value=0.0):
    """
    Applies vectorized cutout to a batch of images.
    
    Args:
        batch (Tensor): Batch of images.
        cutout_frac (float): Fraction of image height to use as cutout size.
        cutout_prob (float): Probability of applying cutout per sample.
        fill_value (float): Value to fill the cut area.
    
    Returns:
        Tensor: Batch after cutout.
    """
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
    return batch * mask + fill_value * (1 - mask)

# ---------------------------
# Data Loading Utilities
# ---------------------------
def load_tensor_datasets(save_dir):
    """
    Loads train, validation, and test TensorDatasets from the given directory.
    Expects filenames: "train_tensorset.pth", "val_tensorset.pth", "test_tensorset.pth".
    
    Args:
        save_dir (str): Directory containing the dataset files.
        
    Returns:
        tuple: (train_tds, val_tds, test_tds)
    """
    tqdm.write(f"[load_tensor_datasets] Loading TensorDatasets from '{save_dir}'...")
    train_tds = torch.load(os.path.join(save_dir, "train_tensorset.pth"))
    val_tds   = torch.load(os.path.join(save_dir, "val_tensorset.pth"))
    test_tds  = torch.load(os.path.join(save_dir, "test_tensorset.pth"))
    tqdm.write(f"[load_tensor_datasets] Loaded: Train={len(train_tds)}, Val={len(val_tds)}, Test={len(test_tds)}")
    return train_tds, val_tds, test_tds

def get_dataloaders(train_tds, val_tds, test_tds, batch_size, shuffle_seed=12345):
    """
    Creates DataLoader objects for the provided TensorDatasets.
    
    Args:
        train_tds, val_tds, test_tds: TensorDatasets.
        batch_size (int): Batch size for loading.
        shuffle_seed (int): Seed for reproducible shuffling.
    
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    tqdm.write(f"[get_dataloaders] Creating DataLoaders with batch_size={batch_size}...")
    g = torch.Generator()
    g.manual_seed(shuffle_seed)
    train_loader = DataLoader(train_tds, batch_size=batch_size, shuffle=True, generator=g)
    val_loader   = DataLoader(val_tds, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_tds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader

# ---------------------------
# Augmentation Pipeline Builder
# ---------------------------
def build_augmentation_pipeline(policy, color_mode='rgb'):
    """
    Constructs an augmentation pipeline as an nn.Sequential based on the provided policy.
    
    Args:
        policy (dict): Dictionary with augmentation parameters.
        color_mode (str): 'rgb' or 'grayscale'. Adjusts color jitter accordingly.
        
    Returns:
        nn.Sequential: The augmentation pipeline.
    """
    ops = []
    
    # Rotation
    if policy.get("rotate_deg", 0) > 0:
        rotate_deg = policy["rotate_deg"]
        rotate_transform = T.RandomRotation(degrees=(-rotate_deg, rotate_deg))
        ops.append(RandomApplyCustom(rotate_transform, p=policy.get("rotate_prob", 0.0)))
    
    # Translation
    if policy.get("translate_frac", 0) > 0:
        translate_transform = T.RandomAffine(degrees=0,
                                               translate=(policy.get("translate_frac", 0.0),
                                                          policy.get("translate_frac", 0.0)))
        ops.append(RandomApplyCustom(translate_transform, p=policy.get("translate_prob", 0.0)))
    
    # Horizontal Flip
    ops.append(RandomApplyCustom(T.RandomHorizontalFlip(p=1.0), p=policy.get("flip_prob", 0.0)))
    
    # Perspective
    if policy.get("perspective_distortion", 0) > 0:
        persp_transform = T.RandomPerspective(distortion_scale=policy.get("perspective_distortion", 0.0), p=1.0)
        ops.append(RandomApplyCustom(persp_transform, p=policy.get("perspective_prob", 0.0)))
    
    # Color Jitter
    if policy.get("color_jitter_strength", 0) > 0:
        cjs = policy["color_jitter_strength"]
        if color_mode == 'rgb':
            color_jitter = T.ColorJitter(brightness=cjs, contrast=cjs, saturation=cjs, hue=0.05)
        else:
            # For grayscale, only brightness and contrast are meaningful.
            color_jitter = T.ColorJitter(brightness=cjs, contrast=cjs)
        ops.append(RandomApplyCustom(color_jitter, p=policy.get("color_jitter_prob", 0.0)))
    
    # Scale / Random Resized Crop
    if policy.get("scale_prob", 0) > 0:
        final_size = (32, 32) if color_mode == 'rgb' else (28, 28)
        scale_transform = T.RandomResizedCrop(final_size,
                                              scale=(0.8, 1.2),
                                              ratio=(0.9, 1.1))
        ops.append(RandomApplyCustom(scale_transform, p=policy.get("scale_prob", 0.0)))
    
    # Gaussian Noise
    if policy.get("noise_std", 0) > 0:
        ops.append(RandomApplyCustom(GaussianNoise(std=policy.get("noise_std", 0.05)),
                                     p=policy.get("noise_prob", 0.0)))
    
    augmentation_pipeline = nn.Sequential(*ops)
    return augmentation_pipeline

# ---------------------------
# Augmented Dataset Creation
# ---------------------------
def create_augmented_dataset_batched(original_dataset, augmentation_pipeline, num_copies, policy, batch_size_augment=2056, device='cpu'):
    """
    Applies the augmentation pipeline in batches over the entire dataset and produces additional copies.
    
    Args:
        original_dataset (TensorDataset): The original dataset.
        augmentation_pipeline (nn.Module): The augmentation pipeline to apply.
        num_copies (int): How many augmented copies to generate.
        policy (dict): Augmentation parameters (used for mixup/cutout).
        batch_size_augment (int): Batch size used during augmentation.
        device (str): Device to perform augmentations on.
        
    Returns:
        TensorDataset: A new dataset including both original and augmented images.
    """
    all_images = []
    all_labels = []
    for i in range(len(original_dataset)):
        img, label = original_dataset[i]
        all_images.append(img.unsqueeze(0))
        all_labels.append(torch.tensor([label]))
    images = torch.cat(all_images, dim=0)  # (N, C, H, W)
    labels = torch.cat(all_labels, dim=0)
    
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
            batch = images[start:end].to(device)
            batch = augmentation_pipeline(batch)
            batch = batched_mixup(batch, mixup_prob, mixup_alpha)
            batch = batched_cutout(batch, cutout_size, cutout_prob, fill_value=cutout_fill)
            aug_images_list.append(batch.cpu())
        aug_batch = torch.cat(aug_images_list, dim=0)
        augmented_batches.append(aug_batch)
        
    final_images = torch.cat(augmented_batches, dim=0)
    final_labels = labels.repeat(num_copies + 1)
    return TensorDataset(final_images, final_labels)

# ---------------------------
# Evaluation of Augmentation Policy
# ---------------------------
def evaluate_augmentation_policy(policy_params, init_state, train_tds, val_tds, test_tds,
                                 model_class, batch_size, num_epochs, device, seed=None, optimizer_params=None):
    """
    Trains a model on an augmented dataset created with the given augmentation policy and returns validation accuracy.
    
    Args:
        policy_params (dict): Augmentation parameters.
        init_state (dict): Initial model state (to reset model weights).
        train_tds, val_tds, test_tds: TensorDatasets.
        model_class: The model class to be instantiated (should accept num_classes parameter).
        batch_size (int): Batch size.
        num_epochs (int): Number of training epochs.
        device (str): Device to run on.
        seed (int, optional): If provided, sets the random seed.
        optimizer_params (dict, optional): Dict with optimizer settings. If None, defaults to AdamW(lr=1e-3).
        
    Returns:
        float: Validation accuracy.
    """
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    
    # Ensure all expected keys are present in the policy.
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
        "color_jitter_prob": policy_params.get("color_jitter_prob", 0.0),
        "color_jitter_strength": policy_params.get("color_jitter_strength", 0.0),
        "scale_prob": policy_params.get("scale_prob", 0.0),
        "mixup_prob": policy_params.get("mixup_prob", 0.0),
        "mixup_alpha": policy_params.get("mixup_alpha", 0.2),
        "cutout_prob": policy_params.get("cutout_prob", 0.0),
        "cutout_size": policy_params.get("cutout_size", 0.0),
        "cutout_fill": policy_params.get("cutout_fill", 0.0)
    }
    num_copies = max(1, int(round(policy_params.get("num_augmented_copies", 1))))
    
    # Determine color mode from dataset channels if not explicitly provided.
    color_mode = 'rgb' if train_tds.tensors[0].shape[1] == 3 else 'grayscale'
    augmentation_pipeline = build_augmentation_pipeline(policy, color_mode=color_mode).to(device)
    augmented_train_tds = create_augmented_dataset_batched(train_tds, augmentation_pipeline, num_copies, policy, device=device)
    
    train_loader, val_loader, _ = get_dataloaders(augmented_train_tds, val_tds, test_tds, batch_size)
    
    num_classes = int(train_tds.tensors[1].max().item() + 1)
    model = model_class(num_classes=num_classes).to(device)
    model.load_state_dict(init_state)
    
    criterion = nn.CrossEntropyLoss()
    if optimizer_params is None:
        optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    else:
        optimizer_class = optimizer_params.get("optimizer_class", optim.AdamW)
        lr = optimizer_params.get("lr", 1e-3)
        optimizer = optimizer_class(model.parameters(), lr=lr,
                                    **{k: v for k, v in optimizer_params.items() if k not in ["optimizer_class", "lr"]})
    
    for epoch in range(num_epochs):
        model.train()
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    
    model.eval()
    total_correct, total_samples = 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)
    accuracy = total_correct / total_samples if total_samples > 0 else 0.0
    return accuracy

# ---------------------------
# Optimization Utility Functions
# ---------------------------
def zero_check(policy_params, threshold=0.01):
    """
    Forces parameters with absolute value below threshold to 0, excluding keys 'num_augmented_copies' and 'mixup_alpha'.
    
    Args:
        policy_params (dict): Augmentation parameters.
        threshold (float): Threshold below which values are zeroed.
        
    Returns:
        dict: Modified policy parameters.
    """
    modified_policy = policy_params.copy()
    for key in policy_params:
        if key not in ["num_augmented_copies", "mixup_alpha"]:
            val = policy_params[key]
            if isinstance(val, (float, int)) and abs(val) < threshold:
                modified_policy[key] = 0.0
    return modified_policy

def sequential_zero_check(initial_best_params, init_state, train_tds, val_tds, test_tds, model_class,
                            param_list, best_val_acc, threshold=0.01, batch_size=1028, num_epochs=5, seed=9999):
    """
    Sequentially tests setting each parameter to 0 (except a few) and keeps the change if validation accuracy improves.
    
    Args:
        initial_best_params (dict): Current best augmentation policy.
        init_state (dict): Initial model state.
        train_tds, val_tds, test_tds: TensorDatasets.
        model_class: The model class.
        param_list (list): List of parameter names to test.
        best_val_acc (float): Current best validation accuracy.
        threshold (float): Minimum threshold for considering a parameter as non-zero.
        batch_size (int): Batch size.
        num_epochs (int): Number of epochs for evaluation.
        seed (int): Random seed.
        
    Returns:
        tuple: (Updated policy parameters, updated best validation accuracy)
    """
    current_params = initial_best_params.copy()
    for key in param_list:
        if key in ["num_augmented_copies", "mixup_alpha"]:
            continue
        if abs(current_params.get(key, 0.0)) < threshold:
            continue
        test_params = current_params.copy()
        test_params[key] = 0.0
        new_val_acc = evaluate_augmentation_policy(test_params, init_state, train_tds, val_tds, test_tds,
                                                   model_class, batch_size, num_epochs, DEVICE, seed=seed)
        if new_val_acc > best_val_acc:
            tqdm.write(f"[sequential_zero_check] Setting '{key}'=0 improved Val Acc {best_val_acc:.4f} -> {new_val_acc:.4f}")
            best_val_acc = new_val_acc
            current_params = test_params
        else:
            tqdm.write(f"[sequential_zero_check] Setting '{key}'=0 did NOT improve (Val Acc: {new_val_acc:.4f} <= {best_val_acc:.4f}), reverting.")
    return current_params, best_val_acc

def check_max_copies(policy_params, pbounds, init_state, train_tds, val_tds, test_tds, model_class,
                     batch_size=1028, num_epochs=5, seed=9999):
    """
    Checks whether using the maximum allowed num_augmented_copies (from pbounds) improves performance.
    
    Args:
        policy_params (dict): Current policy parameters.
        pbounds (dict): The search bounds including 'num_augmented_copies'.
        init_state (dict): Initial model state.
        train_tds, val_tds, test_tds: TensorDatasets.
        model_class: The model class.
        batch_size (int): Batch size.
        num_epochs (int): Number of epochs for evaluation.
        seed (int): Random seed.
        
    Returns:
        tuple: (Potentially updated policy parameters, new validation accuracy or None)
    """
    max_copies = int(pbounds["num_augmented_copies"][1])
    if max_copies != policy_params["num_augmented_copies"]:
        temp_params = policy_params.copy()
        temp_params["num_augmented_copies"] = max_copies
        max_copy_val_acc = evaluate_augmentation_policy(temp_params, init_state, train_tds, val_tds, test_tds,
                                                        model_class, batch_size, num_epochs, DEVICE, seed=seed)
        return temp_params, max_copy_val_acc
    return policy_params, None

# ---------------------------
# Main Pipeline Function
# ---------------------------
def run_augmentation_policy_search(
    dataset_dir,
    results_dir,
    model_class,
    pbounds,
    batch_size=1028,
    num_epochs_proxy=5,
    final_train_epochs=5,
    optimizer_params=None,
    color_mode=None,
    seed=42,
    initial_points=5,
    n_iter=50
):
    """
    Runs the full augmentation policy search pipeline.
    
    Args:
        dataset_dir (str): Directory containing "train_tensorset.pth", "val_tensorset.pth", and "test_tensorset.pth".
        results_dir (str): Directory to save augmented dataset, policy JSON, and logs.
        model_class: Model class to instantiate (should accept num_classes parameter).
        pbounds (dict): Dictionary with parameter bounds for Bayesian Optimization.
        batch_size (int): Batch size for training.
        num_epochs_proxy (int): Number of epochs to train during proxy evaluation.
        final_train_epochs (int): Number of epochs for final training on the augmented dataset.
        optimizer_params (dict, optional): Dictionary with optimizer settings.
        color_mode (str, optional): 'rgb' or 'grayscale'. If None, inferred from dataset.
        seed (int): Random seed.
        initial_points (int): Number of initial random points for Bayesian Optimization.
        n_iter (int): Number of Bayesian Optimization iterations.
    
    Returns:
        dict: Results containing best policy, validation and test accuracies, and file paths.
    """
    os.makedirs(results_dir, exist_ok=True)
    # Load tensor datasets
    train_tds, val_tds, test_tds = load_tensor_datasets(dataset_dir)
    num_classes = int(train_tds.tensors[1].max().item() + 1)
    
    # Set the initial model state for reproducibility.
    torch.manual_seed(seed)
    init_model = model_class(num_classes=num_classes)
    initial_state = copy.deepcopy(init_model.state_dict())
    
    global DEVICE
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Infer color mode if not provided.
    if color_mode is None:
        c = train_tds.tensors[0].shape[1]
        color_mode = 'rgb' if c == 3 else 'grayscale'
    
    # Define objective wrapper for Bayesian Optimization.
    def objective_wrapper(**kwargs):
        kwargs["num_augmented_copies"] = max(1, int(round(kwargs.get("num_augmented_copies", 1))))
        acc = evaluate_augmentation_policy(kwargs, initial_state, train_tds, val_tds, test_tds,
                                           model_class, batch_size, num_epochs_proxy, DEVICE, seed=seed, optimizer_params=optimizer_params)
        tqdm.write(f"[objective_function] Candidate params: {kwargs} => Val Accuracy: {acc:.4f}")
        return acc

    optimizer_bo = BayesianOptimization(
        f=objective_wrapper,
        pbounds=pbounds,
        random_state=seed
    )
    
    tqdm.write("[bayes_opt] Starting augmentation policy search with Bayesian Optimization...")
    optimizer_bo.maximize(init_points=initial_points, n_iter=n_iter)
    
    best_params = optimizer_bo.max["params"]
    best_params["num_augmented_copies"] = int(round(best_params["num_augmented_copies"]))
    best_val_acc = optimizer_bo.max["target"]
    tqdm.write(f"[bayes_opt] Best augmentation parameters found: {best_params} with validation accuracy: {best_val_acc:.4f}")
    
    # Zero-check: Force near-zero parameters to 0.
    zero_params = zero_check(best_params, threshold=0.01)
    zero_val_acc = evaluate_augmentation_policy(zero_params, initial_state, train_tds, val_tds, test_tds,
                                                model_class, batch_size, num_epochs_proxy, DEVICE, seed=seed, optimizer_params=optimizer_params)
    if zero_val_acc > best_val_acc:
        tqdm.write(f"[zero_check] Forcing near-zero values improved Val Acc from {best_val_acc:.4f} to {zero_val_acc:.4f}.")
        best_params = zero_params
        best_val_acc = zero_val_acc
    else:
        tqdm.write(f"[zero_check] Forcing near-zero values did not improve performance (Val Acc: {zero_val_acc:.4f}).")
    
    # Sequential zero-check.
    param_order = list(pbounds.keys())
    updated_params, updated_val_acc = sequential_zero_check(best_params, initial_state, train_tds, val_tds, test_tds,
                                                            model_class, param_order, best_val_acc,
                                                            threshold=0.01, batch_size=batch_size, num_epochs=num_epochs_proxy, seed=seed)
    if updated_val_acc > best_val_acc:
        best_params = updated_params
        best_val_acc = updated_val_acc
    
    # Check if using the maximum number of copies helps.
    best_params, max_copy_val_acc = check_max_copies(best_params, pbounds, initial_state, train_tds, val_tds, test_tds,
                                                     model_class, batch_size=batch_size, num_epochs=num_epochs_proxy, seed=seed)
    if max_copy_val_acc is not None and max_copy_val_acc > best_val_acc:
        best_params = best_params
        best_val_acc = max_copy_val_acc
        tqdm.write(f"[check_max_copies] Using max num_augmented_copies improved Val Acc to {best_val_acc:.4f}.")
    else:
        tqdm.write(f"[check_max_copies] Using max copies did not improve performance.")
    
    tqdm.write(f"[final] Final best augmentation policy: {best_params}")
    tqdm.write(f"[final] Final best validation accuracy: {best_val_acc:.4f}")
    
    # Create final augmented training dataset using the best policy.
    best_augmentation_pipeline = build_augmentation_pipeline(best_params, color_mode=color_mode).to(DEVICE)
    num_copies = best_params["num_augmented_copies"]
    tqdm.write("[final] Creating final augmented training dataset...")
    final_augmented_train_tds = create_augmented_dataset_batched(train_tds, best_augmentation_pipeline, num_copies, best_params, device=DEVICE)
    
    final_save_path = os.path.join(results_dir, "train_augmented_tensorset.pth")
    torch.save(final_augmented_train_tds, final_save_path)
    tqdm.write(f"[final] Final augmented TensorDataset saved to: {final_save_path}")
    
    policy_save_path = os.path.join(results_dir, "best_augmentation_policy.json")
    with open(policy_save_path, "w") as f:
        json.dump(best_params, f, indent=4)
    tqdm.write(f"[final] Best augmentation policy saved to: {policy_save_path}")
    
    # Final model training and test evaluation.
    final_train_loader, _, test_loader = get_dataloaders(final_augmented_train_tds, val_tds, test_tds, batch_size)
    
    final_model = model_class(num_classes=num_classes).to(DEVICE)
    final_model.load_state_dict(initial_state)
    criterion = nn.CrossEntropyLoss()
    if optimizer_params is None:
        optimizer_final = optim.AdamW(final_model.parameters(), lr=1e-3)
    else:
        optimizer_class = optimizer_params.get("optimizer_class", optim.AdamW)
        lr = optimizer_params.get("lr", 1e-3)
        optimizer_final = optimizer_class(final_model.parameters(), lr=lr,
                                          **{k: v for k, v in optimizer_params.items() if k not in ["optimizer_class", "lr"]})
    
    tqdm.write(f"[final] Training final model on augmented dataset for {final_train_epochs} epochs...")
    for epoch in range(final_train_epochs):
        final_model.train()
        for images, labels in tqdm(final_train_loader, desc=f"Final Epoch {epoch+1}/{final_train_epochs}", leave=False):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer_final.zero_grad()
            outputs = final_model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer_final.step()
    
    final_model.eval()
    total_correct, total_samples = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = final_model(images)
            preds = outputs.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)
    final_test_acc = total_correct / total_samples if total_samples > 0 else 0.0
    tqdm.write(f"[final] Final Test Accuracy: {final_test_acc:.4f}")
    
    results = {
        "best_policy": best_params,
        "best_val_acc": best_val_acc,
        "final_test_acc": final_test_acc,
        "augmented_dataset_path": final_save_path,
        "policy_save_path": policy_save_path
    }
    return results

# ---------------------------
# __main__ block for standalone execution
# ---------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Augmentation Policy Search with Bayesian Optimization")
    parser.add_argument("--dataset_dir", type=str, required=True,
                        help="Directory containing train_tensorset.pth, val_tensorset.pth, and test_tensorset.pth.")
    parser.add_argument("--results_dir", type=str, required=True, help="Directory to save results.")
    parser.add_argument("--model", type=str, required=True, help="Model type: 'CNN' or 'EfficientCNN'.")
    parser.add_argument("--batch_size", type=int, default=1028, help="Batch size for training.")
    parser.add_argument("--proxy_epochs", type=int, default=5, help="Number of proxy training epochs.")
    parser.add_argument("--final_epochs", type=int, default=5, help="Number of final training epochs.")
    parser.add_argument("--initial_points", type=int, default=5, help="Initial random points for Bayesian Optimization.")
    parser.add_argument("--n_iter", type=int, default=50, help="Number of Bayesian Optimization iterations.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for reproducibility.")
    args = parser.parse_args()
    
    # Import the specified model from the model module.
    from model import CNN, EfficientCNN
    if args.model == "CNN":
        model_class = CNN
    elif args.model == "EfficientCNN":
        model_class = EfficientCNN
    else:
        raise ValueError("Unknown model type. Choose 'CNN' or 'EfficientCNN'.")
    
    # Define default parameter bounds.
    pbounds = {
        "rotate_prob": (0.3, 0.9),
        "rotate_deg": (0, 25),
        "translate_prob": (0.5, 1.0),
        "translate_frac": (0.0, 0.25),
        "flip_prob": (0.5, 0.9),
        "perspective_prob": (0.3, 0.8),
        "perspective_distortion": (0.0, 0.5),
        "color_jitter_prob": (0.5, 0.9),
        "color_jitter_strength": (0.1, 0.5),
        "noise_prob": (0.2, 0.6),
        "noise_std": (0.0, 0.1),
        "mixup_prob": (0.3, 0.7),
        "mixup_alpha": (0.2, 0.8),
        "cutout_prob": (0.5, 0.9),
        "cutout_size": (0.1, 0.4),
        "cutout_fill": (0.0, 0.3),
        "scale_prob": (0.5, 0.9),
        "num_augmented_copies": (8, 12)
    }
    
    results = run_augmentation_policy_search(
        dataset_dir=args.dataset_dir,
        results_dir=args.results_dir,
        model_class=model_class,
        pbounds=pbounds,
        batch_size=args.batch_size,
        num_epochs_proxy=args.proxy_epochs,
        final_train_epochs=args.final_epochs,
        seed=args.seed,
        initial_points=args.initial_points,
        n_iter=args.n_iter
    )
    
    print("Augmentation Policy Search Complete. Results:")
    print(json.dumps(results, indent=4))
