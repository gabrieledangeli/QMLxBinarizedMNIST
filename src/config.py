import torch

# Random Seed
SEED = 42

# Data Configuration
CLASSES = [0, 1]  # Binary classification for simplicity (e.g., 0 vs 1)
NUM_CLASSES = len(CLASSES)
INPUT_DIM = 784   # 28x28 images flattened

# Training Configuration
BATCH_SIZE = 32
LEARNING_RATE = 0.01
EPOCHS_CLASSICAL = 5
EPOCHS_QUANTUM = 5

# Quantum Configuration
N_QUBITS = 4      # Number of qubits
N_LAYERS = 2      # Number of layers in the quantum circuit

# QNN/HQNN Specific Configuration
QNN_LAYERS = 2    # Layers for QNN (RealAmplitudes)
HQNN_LAYERS = 2   # Layers for HQNN (StronglyEntanglingLayers)
QUANTUM_LR = 0.01 # Learning rate for quantum models
