import numpy as np
import torch
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from . import config

def load_binarized_mnist(n_qubits=None, use_pca=False):
    """
    Loads MNIST data, filters for two classes, and optionally applies PCA.
    
    Args:
        n_qubits (int): Number of components for PCA (if use_pca is True).
        use_pca (bool): Whether to apply PCA dimensionality reduction.
        
    Returns:
        tuple: (x_train, y_train, x_test, y_test) as Torch tensors.
    """
    print("Loading MNIST dataset...")
    # Fetch MNIST data
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
    X = mnist.data
    y = mnist.target.astype(int)

    # Filter for specific classes (binary classification)
    mask = np.isin(y, config.CLASSES)
    X = X[mask]
    y = y[mask]

    # Remap labels to 0 and 1
    y = np.where(y == config.CLASSES[0], 0, 1)

    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=config.SEED, stratify=y
    )

    # Normalize data (StandardScaler)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    if use_pca and n_qubits is not None:
        print(f"Applying PCA to reduce dimensions to {n_qubits}...")
        pca = PCA(n_components=n_qubits)
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)
        
        # Scale features to [0, pi] for AngleEmbedding
        min_val = X_train.min(axis=0)
        max_val = X_train.max(axis=0)
        
        X_train = np.pi * (X_train - min_val) / (max_val - min_val + 1e-8)
        X_test = np.pi * (X_test - min_val) / (max_val - min_val + 1e-8)

    # Convert to PyTorch tensors
    x_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    x_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    print(f"Data loaded. Train shape: {x_train_tensor.shape}, Test shape: {x_test_tensor.shape}")
    return x_train_tensor, y_train_tensor, x_test_tensor, y_test_tensor
