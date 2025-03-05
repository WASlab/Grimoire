import os
import json
import random
import math
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms as T
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

# --- For demonstration, we import the 'CNN' from an external file. ---
# You can replace this with your own model. The script won't retrain it for every candidate
# once we build a SURROGATE. We only do limited runs to create training data for the surrogate.
from model import CNN  # => "model.py" with your custom CNN or other model

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 512
RESULTS_DIR = "./kmnist/surrogate_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# =============== Hyperparameters controlling partial training (only for surrogate data collection) ===============
NUM_EPOCHS_REAL_MODEL = 3  # small partial training to get a rough measure of performance
LEARNING_RATE_REAL_MODEL = 1e-3
SEED = 42
torch.manual_seed(SEED)
random.seed(SEED)

# ---------------------------
# 1) Data Loading
# ---------------------------
def load_tensor_datasets(base_dir="./kmnist/saved_tensorsets"):
    """
    Modify this path or logic for your dataset. We assume you have
    train/val/test as TensorDataset objects stored via torch.save().
    """
    train_tds = torch.load(os.path.join(base_dir, "kmnist_train_tensorset.pth"))
    val_tds = torch.load(os.path.join(base_dir, "kmnist_val_tensorset.pth"))
    test_tds = torch.load(os.path.join(base_dir, "kmnist_test_tensorset.pth"))
    print(f"Loaded: train={len(train_tds)}, val={len(val_tds)}, test={len(test_tds)}")
    return train_tds, val_tds, test_tds

def get_dataloader(ds, batch_size=BATCH_SIZE, shuffle=True):
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

# ---------------------------
# 2) Augmentation Pipeline
# ---------------------------
class RandomApplyCustom(nn.Module):
    """
    Example: randomly apply some transform with probability p.
    Useful for rotation, affine, or noise transforms.
    """
    def __init__(self, transform, p=0.5):
        super().__init__()
        self.transform = transform
        self.p = p

    def forward(self, img):
        if torch.rand(1).item() < self.p:
            return self.transform(img)
        return img

class GaussianNoise(nn.Module):
    """
    Example GPU-friendly Gaussian noise transform.
    """
    def __init__(self, std=0.05):
        super().__init__()
        self.std = std
        
    def forward(self, x):
        return x + torch.randn_like(x) * self.std

def build_augmentation_pipeline(policy):
    """
    Build a pipeline as nn.Sequential for the chosen augmentation hyperparameters.
    policy is a dict, e.g.:
      {
        "rotate_deg": 15,
        "rotate_prob": 0.5,
        "translate_frac": 0.1,
        "translate_prob": 0.2,
        "noise_std": 0.05,
        "noise_prob": 0.3,
        ...
      }
    """
    ops = []
    # Example: rotation
    if policy.get("rotate_deg", 0) > 0:
        deg = policy["rotate_deg"]
        prob = policy.get("rotate_prob", 0.5)
        ops.append(RandomApplyCustom(T.RandomRotation(degrees=(-deg, deg)), p=prob))

    # Example: translation
    if policy.get("translate_frac", 0) > 0:
        tfrac = policy["translate_frac"]
        prob = policy.get("translate_prob", 0.5)
        ops.append(RandomApplyCustom(
            T.RandomAffine(degrees=0, translate=(tfrac, tfrac)),
            p=prob
        ))

    # Example: random horizontal flip
    flip_prob = policy.get("flip_prob", 0.0)
    if flip_prob > 0.0:
        ops.append(RandomApplyCustom(T.RandomHorizontalFlip(p=1.0), p=flip_prob))
    
    # Example: Gaussian noise
    noise_std = policy.get("noise_std", 0.0)
    noise_prob = policy.get("noise_prob", 0.0)
    if noise_std > 0.0 and noise_prob > 0.0:
        ops.append(RandomApplyCustom(GaussianNoise(std=noise_std), p=noise_prob))

    # Add more transforms as needed...
    
    return nn.Sequential(*ops)

def apply_augmentation(dataset, pipeline, n_copies=1):
    """
    Given a dataset of (images, labels), apply pipeline to produce additional copies.
    Returns a new TensorDataset. 
    For large datasets, do it in chunks if memory is an issue.
    """
    all_images, all_labels = [], []
    for img, lab in dataset:
        all_images.append(img.unsqueeze(0))
        all_labels.append(torch.tensor(lab).unsqueeze(0))

    images = torch.cat(all_images, dim=0)  # shape [N, C, H, W]
    labels = torch.cat(all_labels, dim=0)  # shape [N]

    augmented_sets = [images]  # always include the original images
    for _ in range(n_copies):
        # We can do batch augmentation for speed
        # Move entire batch to GPU
        batch = images.to(DEVICE)
        aug_batch = pipeline(batch)
        # Move back to CPU
        augmented_sets.append(aug_batch.cpu())

    final_images = torch.cat(augmented_sets, dim=0)
    final_labels = labels.repeat(n_copies + 1)
    return TensorDataset(final_images, final_labels)

# ---------------------------
# 3) Partial Training of Real Model to get Performance
# ---------------------------
def _train_and_eval_real_model(policy, init_state, train_ds, val_ds, objective="accuracy"):
    """
    We do a short/partial training of the real model on an augmented dataset 
    to measure performance. This is expensive, so we only do it for some policies 
    to gather training data for the surrogate.

    objective can be "accuracy", "macro_f1", etc.
    """
    # Build augmentation
    pipeline = build_augmentation_pipeline(policy).to(DEVICE)
    # Possibly replicate data
    n_copies = int(round(policy.get("num_aug_copies", 1)))
    # Create augmented dataset
    augmented_train_ds = apply_augmentation(train_ds, pipeline, n_copies=n_copies)

    # Real model
    num_classes = int(train_ds.tensors[1].max().item() + 1)
    model = CNN(num_classes=num_classes).to(DEVICE)
    model.load_state_dict(init_state)  # same initialization for fair comparison

    train_loader = get_dataloader(augmented_train_ds, BATCH_SIZE)
    val_loader = get_dataloader(val_ds, BATCH_SIZE, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE_REAL_MODEL)

    # Quick partial training 
    model.train()
    for epoch in range(NUM_EPOCHS_REAL_MODEL):
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    # Evaluate on validation set
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            logits = model(images)
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    if objective == "accuracy":
        score = accuracy_score(all_labels, all_preds)
    elif objective == "macro_f1":
        score = f1_score(all_labels, all_preds, average="macro")
    else:
        # fallback to accuracy
        score = accuracy_score(all_labels, all_preds)
    return score

# ---------------------------
# 4) Surrogate Model
# ---------------------------
class SurrogateNet(nn.Module):
    """
    A small feedforward net that predicts performance from augmentation hyperparams.
    E.g. input vector [rotate_prob, rotate_deg, noise_prob, noise_std, ...].
    """
    def __init__(self, input_dim, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    def forward(self, x):
        # x shape: [batch_size, input_dim]
        return self.net(x)

def train_surrogate(X, y, val_ratio=0.2, epochs=50, lr=1e-3):
    """
    Train a SurrogateNet on the dataset (X -> y).
    X is a torch.Tensor shape [N, D].
    y is shape [N].
    """
    # Split for validation
    N = X.size(0)
    idxs = list(range(N))
    random.shuffle(idxs)
    val_size = int(N * val_ratio)
    val_idxs = idxs[:val_size]
    train_idxs = idxs[val_size:]
    
    X_train = X[train_idxs]
    y_train = y[train_idxs]
    X_val   = X[val_idxs]
    y_val   = y[val_idxs]

    input_dim = X.size(1)
    model = SurrogateNet(input_dim).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # Dataloaders
    train_ds = TensorDataset(X_train, y_train)
    val_ds   = TensorDataset(X_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=64, shuffle=False)

    for ep in range(epochs):
        model.train()
        for bx, by in train_loader:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            optimizer.zero_grad()
            pred = model(bx).squeeze()
            loss = criterion(pred, by)
            loss.backward()
            optimizer.step()
        
        # Evaluate
        model.eval()
        with torch.no_grad():
            val_losses = []
            for bx, by in val_loader:
                bx, by = bx.to(DEVICE), by.to(DEVICE)
                pred = model(bx).squeeze()
                val_losses.append(criterion(pred, by).item())
            val_loss = sum(val_losses)/len(val_losses)
        if (ep+1) % 10 == 0:
            print(f"Epoch {ep+1}/{epochs}, Val MSE: {val_loss:.4f}")
    return model

# ---------------------------
# Main pipeline
# ---------------------------
if __name__ == "__main__":
    # 1) Load data
    train_tds, val_tds, test_tds = load_tensor_datasets("./kmnist/saved_tensorsets")
    # We'll use the same random init for each partial training
    num_classes = int(train_tds.tensors[1].max().item() + 1)
    base_model = CNN(num_classes=num_classes)
    init_state = copy.deepcopy(base_model.state_dict())

    # 2) Collect training data for surrogate
    #    We'll sample random augmentation policies, do partial training, measure performance.
    #    In a real scenario, use a bigger sample_count to cover the space better.
    sample_count = 20
    all_policies = []
    all_scores = []

    for i in range(sample_count):
        # Example random policy
        policy = {
            "rotate_deg": random.uniform(0, 15),
            "rotate_prob": random.uniform(0, 1),
            "translate_frac": random.uniform(0, 0.3),
            "translate_prob": random.uniform(0, 1),
            "noise_std": random.uniform(0, 0.1),
            "noise_prob": random.uniform(0, 1),
            "num_aug_copies": random.randint(1, 3),
            "flip_prob": 0.0,    # for example, keep it off if your dataset is already symmetrical
        }

        # Evaluate partial training performance
        score = _train_and_eval_real_model(
            policy=policy, 
            init_state=init_state, 
            train_ds=train_tds, 
            val_ds=val_tds, 
            objective="accuracy"
        )
        print(f"[DataCollection] Policy={policy}, Score={score:.4f}")

        # Store in lists
        all_policies.append(policy)
        all_scores.append(score)

    # 3) Convert policies -> vector representation for Surrogate
    #    Suppose we have 6 numeric parameters: rotate_deg, rotate_prob, translate_frac, translate_prob, noise_std, noise_prob
    #    plus num_aug_copies
    #    (flip_prob is also numeric, or you can skip it if it's always zero.)
    def policy_to_vector(pol):
        return [
            pol["rotate_deg"],
            pol["rotate_prob"],
            pol["translate_frac"],
            pol["translate_prob"],
            pol["noise_std"],
            pol["noise_prob"],
            pol["num_aug_copies"],
            pol["flip_prob"],
        ]

    X_list = []
    for p in all_policies:
        X_list.append(policy_to_vector(p))

    X = torch.tensor(X_list, dtype=torch.float32)
    y = torch.tensor(all_scores, dtype=torch.float32)

    # 4) Train Surrogate
    surrogate = train_surrogate(X, y, val_ratio=0.2, epochs=50, lr=1e-3)

    # 5) Evaluate Surrogate on entire dataset (train+val) just to see R^2 or MSE
    surrogate.eval()
    with torch.no_grad():
        pred_all = surrogate(X.to(DEVICE)).squeeze().cpu().numpy()

    # R^2 or correlation
    from sklearn.metrics import r2_score, mean_squared_error
    r2 = r2_score(y.numpy(), pred_all)
    mse = mean_squared_error(y.numpy(), pred_all)
    print(f"Surrogate R^2: {r2:.3f}, MSE: {mse:.4f}")

    # 6) (Optional) Surrogate-based search for best augmentation
    #    For demonstration, let's do a naive random search on the surrogate:
    best_pred_score = -1e9
    best_policy = None
    for _ in range(1000):
        pol = {
            "rotate_deg": random.uniform(0, 15),
            "rotate_prob": random.uniform(0, 1),
            "translate_frac": random.uniform(0, 0.3),
            "translate_prob": random.uniform(0, 1),
            "noise_std": random.uniform(0, 0.1),
            "noise_prob": random.uniform(0, 1),
            "num_aug_copies": random.randint(1, 5),
            "flip_prob": 0.0
        }
        # Convert to vector
        xv = torch.tensor([policy_to_vector(pol)], dtype=torch.float32, device=DEVICE)
        with torch.no_grad():
            pred_perf = surrogate(xv).item()
        if pred_perf > best_pred_score:
            best_pred_score = pred_perf
            best_policy = pol

    print(f"\n--- Surrogate-based best augmentation policy (predicted score={best_pred_score:.4f}) ---")
    print(best_policy)

    # 7) Final check: Actually measure the performance for that best policy
    real_perf = _train_and_eval_real_model(
        policy=best_policy,
        init_state=init_state,
        train_ds=train_tds,
        val_ds=val_tds,
        objective="accuracy"
    )
    print(f"Real performance of that policy: {real_perf:.4f}")
