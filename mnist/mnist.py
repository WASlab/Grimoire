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
        # Its shape is (1, 28, 28) for MNIST.
        images.append(img.unsqueeze(0))  # Add an extra batch dimension: (1, 1, 28, 28)
        labels.append(lbl)

    # Concatenate along the batch dimension
    images = torch.cat(images, dim=0)  # Shape: (N, 1, 28, 28)
    labels = torch.tensor(labels, dtype=torch.long)  # Shape: (N,)

    return TensorDataset(images, labels)

def create_tensorsets_MNIST(
    data_dir="./data",
    val_size=0.16666,
    random_seed=42,
    save_dir="./saved_tensorsets"
):
    """
    1. Download MNIST.
    2. Split training set into train+val with class balance.
    3. Convert train, val, and test subsets into TensorDatasets.
    4. Save them to disk so we can later create DataLoaders with flexible batch sizes.
    """

    # --- 1. Define transforms ---
    transform = transforms.Compose([transforms.ToTensor()])

    # --- 2. Load entire MNIST training set & test set ---
    MNIST_full = datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transform
    )
    MNIST_test = datasets.MNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=transform
    )

    print(f"Full training set size: {len(MNIST_full)}")
    print(f"Test set size:          {len(MNIST_test)}")

    # --- 3. Create class-balanced train/val split ---
    labels = [MNIST_full[i][1] for i in range(len(MNIST_full))]
    train_indices, val_indices = train_test_split(
        range(len(MNIST_full)),
        test_size=val_size,
        stratify=labels,
        random_state=random_seed
    )

    MNIST_train = Subset(MNIST_full, train_indices)
    MNIST_val   = Subset(MNIST_full, val_indices)

    print(f"Train subset size:      {len(MNIST_train)}")
    print(f"Validation subset size: {len(MNIST_val)}")

    # --- 4. Convert train/val/test splits into TensorDataset ---
    train_tds = to_tensor_dataset(MNIST_train)  # => TensorDataset of shape (N_train, 1, 28, 28)
    val_tds   = to_tensor_dataset(MNIST_val)    # => TensorDataset of shape (N_val,   1, 28, 28)

    # We also convert the official test set
    test_tds  = to_tensor_dataset(MNIST_test)

    # --- 5. Save the TensorDatasets ---
    os.makedirs(save_dir, exist_ok=True)
    torch.save(train_tds, f"{save_dir}/MNIST_train_tensorset.pth")
    torch.save(val_tds,   f"{save_dir}/MNIST_val_tensorset.pth")
    torch.save(test_tds,  f"{save_dir}/MNIST_test_tensorset.pth")

    print(f"TensorDatasets saved in: {save_dir}")

if __name__ == "__main__":
    create_tensorsets_MNIST(
        data_dir="./data",
        val_size=0.16666,
        random_seed=42,
        save_dir="./saved_tensorsets"
    )
