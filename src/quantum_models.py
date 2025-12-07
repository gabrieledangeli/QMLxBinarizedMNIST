import pennylane as qml
import torch
import torch.nn as nn
import numpy as np

class QNNModel(nn.Module):
    """
    Quantum Neural Network (QNN) model inspired by the paper.
    Uses ZZFeatureMap for encoding and RealAmplitudes for the ansatz.
    """
    def __init__(self, n_qubits, num_classes, n_layers):
        super().__init__()
        self.n_qubits = n_qubits
        self.num_classes = num_classes
        
        # Pre-processing embedding (Classical)
        # Maps input features to n_qubits for the quantum circuit
        # In the paper, this might be trained separately, but here we include it 
        # as a simple linear map to ensure dimensions match if input_dim != n_qubits.
        # If input is already PCA-reduced to n_qubits, this could be Identity.
        # For generality, we use a Linear layer.
        self.pre_net = nn.Linear(n_qubits, n_qubits) 
        
        dev = qml.device("lightning.qubit", wires=n_qubits)

        @qml.qnode(dev, interface="torch")
        def qnode(inputs, weights):
            # ZZFeatureMap-like encoding
            # 1. Hadamard on all wires
            for i in range(n_qubits):
                qml.Hadamard(wires=i)
            
            # 2. RZ rotations encoding data
            # Note: inputs are expected to be scaled appropriately (e.g. [0, 2pi])
            for i in range(n_qubits):
                qml.RZ(2 * inputs[..., i], wires=i)
            
            # 3. ZZ interactions (CNOT -> RZ -> CNOT)
            for i in range(n_qubits - 1):
                qml.CNOT(wires=[i, i+1])
                qml.RZ(2 * (np.pi - inputs[..., i]) * (np.pi - inputs[..., i+1]), wires=i+1)
                qml.CNOT(wires=[i, i+1])

            # RealAmplitudes-like ansatz with linear entanglement
            for l in range(n_layers):
                # RY rotations on all qubits
                for i in range(n_qubits):
                    qml.RY(weights[l, i], wires=i)
                
                # Linear entanglement
                for i in range(n_qubits - 1):
                    qml.CNOT(wires=[i, i+1])
            
            # Final rotations
            for i in range(n_qubits):
                qml.RY(weights[n_layers, i], wires=i)

            # Measurement: Expectation of Z on all qubits
            return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]

        # Weight shapes: (n_layers + 1, n_qubits) for RealAmplitudes (RY rotations)
        weight_shapes = {"weights": (n_layers + 1, n_qubits)}
        self.qlayer = qml.qnn.TorchLayer(qnode, weight_shapes)
        
        # Final classical layer to map expectation values to classes
        self.post_net = nn.Linear(n_qubits, num_classes)

    def forward(self, x):
        x = self.pre_net(x)
        x = self.qlayer(x)
        x = self.post_net(x)
        return x

class HQNNModel(nn.Module):
    """
    Hybrid Quantum Neural Network (HQNN) model inspired by the paper.
    Uses a trainable classical embedding and a quantum circuit with QAOAEmbedding 
    and StronglyEntanglingLayers.
    """
    def __init__(self, input_dim, n_qubits, num_classes, n_layers):
        super().__init__()
        
        # Classical Embedding Layer (Trainable)
        # Maps high-dimensional input (e.g. 784) to n_qubits
        self.embedding = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, n_qubits),
            nn.Tanh() # Normalize to [-1, 1] or similar for quantum encoding
        )
        
        dev = qml.device("lightning.qubit", wires=n_qubits)

        @qml.qnode(dev, interface="torch")
        def qnode_split(inputs, w_qaoa, w_strong):
            qml.QAOAEmbedding(features=inputs, weights=w_qaoa, wires=range(n_qubits))
            qml.StronglyEntanglingLayers(weights=w_strong, wires=range(n_qubits), ranges=[1]*n_layers, imprimitive=qml.CNOT)
            return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]
            
        qaoa_layers = 1
        strong_layers = n_layers
        
        # Shape for QAOAEmbedding (assuming local_field='z' default)
        # It takes features and weights. weights shape: (layers, 2 * num_wires)
        shape_qaoa = (qaoa_layers, 2 * n_qubits)
        
        # Shape for StronglyEntanglingLayers
        shape_strong = (strong_layers, n_qubits, 3)
        
        weight_shapes = {
            "w_qaoa": shape_qaoa,
            "w_strong": shape_strong
        }
        
        self.qlayer = qml.qnn.TorchLayer(qnode_split, weight_shapes)
        self.post_net = nn.Linear(n_qubits, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        # Scale to appropriate range for QAOA if needed, Tanh gives [-1, 1]
        # QAOA usually expects angles, so maybe scale by pi
        x = x * np.pi 
        x = self.qlayer(x)
        x = self.post_net(x)
        return x

def build_quantum_model(model_type, n_qubits, num_classes, config):
    """
    Factory function to build the specified quantum model.
    """
    if model_type == "qnn":
        # QNN uses PCA features (n_qubits dim)
        return QNNModel(n_qubits, num_classes, config.QNN_LAYERS)
    elif model_type == "hqnn":
        # HQNN uses full input features (784 dim) and learns embedding
        return HQNNModel(config.INPUT_DIM, n_qubits, num_classes, config.HQNN_LAYERS)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
