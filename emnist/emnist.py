# create_tensordatasets.py
import torch
from torch.utils.data import Subset, TensorDataset
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
import os

def to_tensor_dataset(dataset):
    """
    Convert a PyTorch dataset of (image, label) pairs
    to a single TensorDataset containing all images and labels in memory.
    """
    images = []
    labels = []
    for i in range(len(dataset)):
        img, lbl = dataset[i]
        # img is already a torch.Tensor if we used transforms.ToTensor().
        # Its shape is (1, 28, 28) for EMNIST.
        images.append(img.unsqueeze(0))  # Add an extra batch dimension: (1, 1, 28, 28)
        labels.append(lbl)

    # Concatenate along the batch dimension
    images = torch.cat(images, dim=0)  # Shape: (N, 1, 28, 28)
    labels = torch.tensor(labels, dtype=torch.long)  # Shape: (N,)

    return TensorDataset(images, labels)

def create_tensorsets_EMNIST(
    split="byclass",
    data_dir="./data",
    val_size=0.16666,
    random_seed=42,
    save_dir="./saved_tensorsets"
):
    """
    1. Download EMNIST.
    2. Split training set into train+val with class balance.
    3. Convert train, val, and test subsets into TensorDatasets.
    4. Save them to disk so we can later create DataLoaders with flexible batch sizes.
    """

    # --- 1. Define transforms ---
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                                                         )])
    

    # --- 2. Load entire EMNIST training set & test set ---
    EMNIST_full = datasets.EMNIST(
        split="byclass",
        root=data_dir,
        train=True,
        download=True,
        transform=transform
    )
    EMNIST_test = datasets.EMNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=transform
    )

    print(f"Full training set size: {len(EMNIST_full)}")
    print(f"Test set size:          {len(EMNIST_test)}")

    # --- 3. Create class-balanced train/val split ---
    labels = [EMNIST_full[i][1] for i in range(len(EMNIST_full))]
    train_indices, val_indices = train_test_split(
        range(len(EMNIST_full)),
        test_size=val_size,
        stratify=labels,
        random_state=random_seed
    )

    EMNIST_train = Subset(EMNIST_full, train_indices)
    EMNIST_val   = Subset(EMNIST_full, val_indices)

    print(f"Train subset size:      {len(EMNIST_train)}")
    print(f"Validation subset size: {len(EMNIST_val)}")

    # --- 4. Convert train/val/test splits into TensorDataset ---
    train_tds = to_tensor_dataset(EMNIST_train)  # => TensorDataset of shape (N_train, 1, 28, 28)
    val_tds   = to_tensor_dataset(EMNIST_val)    # => TensorDataset of shape (N_val,   1, 28, 28)

    # We also convert the official test set
    test_tds  = to_tensor_dataset(EMNIST_test)

    # --- 5. Save the TensorDatasets ---
    os.makedirs(save_dir, exist_ok=True)
    torch.save(train_tds, f"{save_dir}/EMNIST_train_tensorset.pth")
    torch.save(val_tds,   f"{save_dir}/EMNIST_val_tensorset.pth")
    torch.save(test_tds,  f"{save_dir}/EMNIST_test_tensorset.pth")

    print(f"TensorDatasets saved in: {save_dir}")

if __name__ == "__main__":
    create_tensorsets_EMNIST(
        split="byclass",
        data_dir="./data",
        val_size=0.16666,
        random_seed=42,
        save_dir="./saved_tensorsets"
    )
