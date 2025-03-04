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
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
from torchvision import transforms as T
from tqdm import tqdm
from bayes_opt import BayesianOptimization  # pip install bayesian-optimization

plt.style.use('seaborn-v0_8-darkgrid')

# =============== Global Settings & Hyperparameters ===============
NUM_EPOCHS_PROXY      = 5           # proxy training epochs for objective evaluation
NUM_EPOCHS_FINAL      = 10          # not used here directly, final training uses 5 epochs
FINAL_TRAIN_EPOCHS    = 5           # final training epochs for evaluation on test set
BATCH_SIZE            = 1028
DEVICE                = "cuda" if torch.cuda.is_available() else "cpu"
SAVED_TENSORSETS_DIR  = "./kmnist/saved_tensorsets"
RESULTS_DIR           = "./kmnist/results_augmentation_search"
os.makedirs(RESULTS_DIR, exist_ok=True)
DATA_LOADER_SEED      = 12345      # fixed seed for reproducibility

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
# ---------------------------
def batched_cutout(batch, cutout_frac, cutout_prob):
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
    return batch * mask

# ---------------------------
# Data Loading Functions (assumes tensor datasets already exist)
# ---------------------------
def load_tensor_datasets(save_dir=SAVED_TENSORSETS_DIR):
    tqdm.write(f"[load_tensor_datasets] Loading TensorDatasets from '{save_dir}'...")
    train_tds = torch.load(os.path.join(save_dir, "kmnist_train_tensorset.pth"))
    val_tds   = torch.load(os.path.join(save_dir, "kmnist_val_tensorset.pth"))
    test_tds  = torch.load(os.path.join(save_dir, "kmnist_test_tensorset.pth"))
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
# CNN Architecture (Proxy model)
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
# Augmentation Pipeline Builder (GPU-accelerated)
# ---------------------------
def build_augmentation_pipeline(policy):
    """
    Build a GPU-accelerated augmentation pipeline as nn.Sequential from a given policy dictionary.
    Note: Since images are grayscale, we skip ColorJitter.
    Expected keys (besides mixup and cutout):
      - rotate_prob, rotate_deg, translate_prob, translate_frac, flip_prob,
        noise_prob, noise_std, perspective_prob, perspective_distortion.
    """
    ops = []
    ops.append(RandomApplyCustom(T.RandomRotation(degrees=(-policy["rotate_deg"], policy["rotate_deg"])), p=policy["rotate_prob"]))
    ops.append(RandomApplyCustom(T.RandomAffine(degrees=0, translate=(policy["translate_frac"], policy["translate_frac"])), p=policy["translate_prob"]))
    ops.append(RandomApplyCustom(T.RandomHorizontalFlip(p=1.0), p=policy["flip_prob"]))
    ops.append(RandomApplyCustom(T.RandomPerspective(distortion_scale=policy["perspective_distortion"], p=1.0), p=policy["perspective_prob"]))
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
    labels = torch.cat(all_labels, dim=0)    # (N,)
    
    augmented_batches = [images]  # include original images
    N = images.shape[0]
    for _ in range(num_copies):
        aug_images_list = []
        for start in range(0, N, batch_size_augment):
            end = start + batch_size_augment
            batch = images[start:end].to(DEVICE)
            batch = augmentation_pipeline(batch)
            batch = batched_mixup(batch, policy.get("mixup_prob", 0.0), policy.get("mixup_alpha", 0.2))
            batch = batched_cutout(batch, policy.get("cutout_size", 0.0), policy.get("cutout_prob", 0.0))
            aug_images_list.append(batch.cpu())
        aug_batch = torch.cat(aug_images_list, dim=0)
        augmented_batches.append(aug_batch)
    final_images = torch.cat(augmented_batches, dim=0)
    final_labels = labels.repeat(num_copies + 1)
    return TensorDataset(final_images, final_labels)

# ---------------------------
# Proxy Evaluation Function for Augmentation Policy
# ---------------------------
def evaluate_augmentation_policy(policy_params, init_state, train_tds, val_tds, test_tds,
                                   batch_size=BATCH_SIZE, num_epochs=NUM_EPOCHS_PROXY):
    policy = {
        "rotate_prob": policy_params["rotate_prob"],
        "rotate_deg": policy_params["rotate_deg"],
        "translate_prob": policy_params["translate_prob"],
        "translate_frac": policy_params["translate_frac"],
        "flip_prob": policy_params["flip_prob"],
        "noise_prob": policy_params["noise_prob"],
        "noise_std": policy_params["noise_std"],
        "perspective_prob": policy_params["perspective_prob"],
        "perspective_distortion": policy_params["perspective_distortion"],
        "mixup_prob": policy_params.get("mixup_prob", 0.0),
        "mixup_alpha": policy_params.get("mixup_alpha", 0.2),
        "cutout_prob": policy_params.get("cutout_prob", 0.0),
        "cutout_size": policy_params.get("cutout_size", 0.0)
    }
    num_copies = max(1, int(round(policy_params["num_augmented_copies"])))
    augmentation_pipeline = build_augmentation_pipeline(policy).to(DEVICE)
    augmented_train_tds = create_augmented_dataset_batched(train_tds, augmentation_pipeline, num_copies, policy)
    train_loader, val_loader, test_loader = get_dataloaders(augmented_train_tds, val_tds, test_tds, batch_size)
    
    num_classes = int(train_tds.tensors[1].max().item() + 1)
    model = CNN(num_classes=num_classes).to(DEVICE)
    model.load_state_dict(init_state)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    
    for epoch in range(num_epochs):
        model.train()
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    
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
# ---------------------------
def objective_function(**kwargs):
    kwargs["num_augmented_copies"] = max(1, int(round(kwargs["num_augmented_copies"])))
    acc = evaluate_augmentation_policy(kwargs, initial_state, train_tds, val_tds, test_tds,
                                       batch_size=BATCH_SIZE, num_epochs=NUM_EPOCHS_PROXY)
    tqdm.write(f"[objective_function] Candidate params: {kwargs} => Val Accuracy: {acc:.4f}")
    return acc

# ---------------------------
# Function to force near-zero values to 0 for selected parameters
# ---------------------------
def zero_check(policy_params, threshold=0.01):
    """
    Force parameters that are below the threshold to 0.
    We exclude 'num_augmented_copies' and 'mixup_alpha' from this check.
    """
    modified_policy = policy_params.copy()
    for key in policy_params:
        if key not in ["num_augmented_copies", "mixup_alpha"]:
            if isinstance(policy_params[key], (float, int)) and policy_params[key] < threshold:
                modified_policy[key] = 0.0
    return modified_policy

# ---------------------------
# Main Pipeline: Augmentation Policy Search, Final Dataset, and Final Model Training/Evaluation
# ---------------------------
if __name__ == "__main__":
    train_tds, val_tds, test_tds = load_tensor_datasets(SAVED_TENSORSETS_DIR)
    num_classes = int(train_tds.tensors[1].max().item() + 1)
    torch.manual_seed(42)
    init_model = CNN(num_classes=num_classes)
    initial_state = copy.deepcopy(init_model.state_dict())
    
    # Define search space (for grayscale images, skipping color jitter)
    pbounds = {
        "rotate_prob": (0.0, 1.0),
        "rotate_deg": (0, 15),
        "translate_prob": (0.0, 1.0),
        "translate_frac": (0, 0.3),
        "flip_prob": (0.0, 0.0),
        "noise_prob": (0.0, 1.0),
        "noise_std": (0, 0.1),
        "perspective_prob": (0.0, 1.0),
        "perspective_distortion": (0, 0.3),
        "num_augmented_copies": (1, 5),
        "mixup_prob": (0.0, 1.0),
        "mixup_alpha": (0.1, 1.0),
        "cutout_prob": (0.0, 1.0),
        "cutout_size": (0.0, 0.5)
    }
    
    optimizer = BayesianOptimization(
        f=objective_function,
        pbounds=pbounds,
        random_state=42,
    )
    
    tqdm.write("[bayes_opt] Starting augmentation policy search with Bayesian Optimization...")
    optimizer.maximize(init_points=5, n_iter=50)
    
    best_params = optimizer.max["params"]
    best_params["num_augmented_copies"] = int(round(best_params["num_augmented_copies"]))
    best_val_acc = optimizer.max["target"]
    tqdm.write(f"[bayes_opt] Best augmentation parameters found: {best_params} with validation accuracy: {best_val_acc:.4f}")
    
    # --------- Final Zero-Check: Force near-zero parameters to 0 and re-evaluate ---------
    zero_params = zero_check(best_params, threshold=0.01)
    zero_val_acc = evaluate_augmentation_policy(zero_params, initial_state, train_tds, val_tds, test_tds,
                                                batch_size=BATCH_SIZE, num_epochs=NUM_EPOCHS_PROXY)
    if zero_val_acc > best_val_acc:
        tqdm.write(f"[zero_check] Forcing near-zero values to 0 improved validation accuracy from {best_val_acc:.4f} to {zero_val_acc:.4f}. Updating best parameters.")
        best_params = zero_params
        best_val_acc = zero_val_acc
    else:
        tqdm.write(f"[zero_check] Forcing near-zero values did not improve performance (Val Accuracy: {zero_val_acc:.4f}). Keeping original parameters.")
    
    # Create final augmented training dataset using batched processing.
    best_augmentation_pipeline = build_augmentation_pipeline(best_params).to(DEVICE)
    num_copies = best_params["num_augmented_copies"]
    tqdm.write("[final] Creating final augmented training dataset with the best augmentation policy...")
    final_augmented_train_tds = create_augmented_dataset_batched(train_tds, best_augmentation_pipeline, num_copies, best_params)
    
    # Save the final augmented dataset and policy parameters.
    final_save_path = os.path.join(RESULTS_DIR, "kmnist_train_augmented_tensorset.pth")
    torch.save(final_augmented_train_tds, final_save_path)
    tqdm.write(f"[final] Final augmented TensorDataset saved to: {final_save_path}")
    
    policy_save_path = os.path.join(RESULTS_DIR, "best_augmentation_policy.json")
    with open(policy_save_path, "w") as f:
        json.dump(best_params, f, indent=4)
    tqdm.write(f"[final] Best augmentation policy saved to: {policy_save_path}")
    
    # ======= FINAL MODEL TRAINING AND TEST EVALUATION =======
    # Create dataloaders for final training and testing.
    final_train_loader, _, test_loader = get_dataloaders(final_augmented_train_tds, val_tds, test_tds, batch_size=BATCH_SIZE)
    
    # Initialize final model with same initial state.
    final_model = CNN(num_classes=num_classes).to(DEVICE)
    final_model.load_state_dict(initial_state)
    criterion = nn.CrossEntropyLoss()
    optimizer_final = optim.AdamW(final_model.parameters(), lr=1e-3)
    
    tqdm.write("[final] Training final model on augmented training dataset for 5 epochs...")
    for epoch in range(FINAL_TRAIN_EPOCHS):
        final_model.train()
        for images, labels in tqdm(final_train_loader, desc=f"Final Epoch {epoch+1}/{FINAL_TRAIN_EPOCHS}"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer_final.zero_grad()
            outputs = final_model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer_final.step()
    
    # Evaluate final model on test set.
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
