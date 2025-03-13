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
        # Its shape is (1, 28, 28) for KMNIST.
        images.append(img.unsqueeze(0))  # Add an extra batch dimension: (1, 1, 28, 28)
        labels.append(lbl)

    # Concatenate along the batch dimension
    images = torch.cat(images, dim=0)  # Shape: (N, 1, 28, 28)
    labels = torch.tensor(labels, dtype=torch.long)  # Shape: (N,)

    return TensorDataset(images, labels)

def create_tensorsets_kmnist(
    data_dir="./data",
    val_size=0.16666,
    random_seed=42,
    save_dir="./saved_tensorsets"
):
    """
    1. Download KMNIST.
    2. Compute the mean and standard deviation on the training set.
    3. Split training set into train+val with class balance.
    4. Convert train, val, and test subsets into TensorDatasets.
    5. Save them to disk so we can later create DataLoaders with flexible batch sizes.
    """
    
    # --- 1. Define a basic transform for computing statistics (only ToTensor)
    basic_transform = transforms.Compose([transforms.ToTensor()])

    # --- 2. Load the full KMNIST training set with the basic transform
    kmnist_full = datasets.KMNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=basic_transform
    )

    # --- 3. Compute the mean and standard deviation on the training set
    # Each image in KMNIST is (1, 28, 28). We stack them and compute channel-wise stats.
    all_images = torch.stack([img for img, _ in kmnist_full])  # Shape: (N, 1, 28, 28)
    mean = all_images.mean(dim=[0, 2, 3])
    std = all_images.std(dim=[0, 2, 3])
    print(f"Computed mean: {mean}")
    print(f"Computed std:  {std}")

    # --- 4. Define a new transform with normalization
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean.tolist(), std.tolist())
    ])

    # --- 5. Reload the KMNIST training and test sets with the new transform
    kmnist_full = datasets.KMNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transform
    )
    kmnist_test = datasets.KMNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=transform
    )

    print(f"Full training set size: {len(kmnist_full)}")
    print(f"Test set size:          {len(kmnist_test)}")

    # --- 6. Create a class-balanced train/validation split
    labels = [kmnist_full[i][1] for i in range(len(kmnist_full))]
    train_indices, val_indices = train_test_split(
        range(len(kmnist_full)),
        test_size=val_size,
        stratify=labels,
        random_state=random_seed
    )

    kmnist_train = Subset(kmnist_full, train_indices)
    kmnist_val   = Subset(kmnist_full, val_indices)

    print(f"Train subset size:      {len(kmnist_train)}")
    print(f"Validation subset size: {len(kmnist_val)}")

    # --- 7. Convert train, validation, and test splits into TensorDatasets
    train_tds = to_tensor_dataset(kmnist_train)  # TensorDataset of shape (N_train, 1, 28, 28)
    val_tds   = to_tensor_dataset(kmnist_val)    # TensorDataset of shape (N_val,   1, 28, 28)
    test_tds  = to_tensor_dataset(kmnist_test)

    # --- 8. Save the TensorDatasets to disk
    os.makedirs(save_dir, exist_ok=True)
    torch.save(train_tds, f"{save_dir}/kmnist_train_tensorset.pth")
    torch.save(val_tds,   f"{save_dir}/kmnist_val_tensorset.pth")
    torch.save(test_tds,  f"{save_dir}/kmnist_test_tensorset.pth")

    print(f"TensorDatasets saved in: {save_dir}")

if __name__ == "__main__":
    create_tensorsets_kmnist(
        data_dir="./data",
        val_size=0.16666,
        random_seed=42,
        save_dir="./saved_tensorsets"
    )
