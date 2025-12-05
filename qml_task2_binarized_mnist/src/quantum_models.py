import pennylane as qml
import torch
import torch.nn as nn

def build_quantum_model(n_qubits, num_classes, config):
    """
    Builds a hybrid quantum-classical model using PennyLane and PyTorch.
    """
    dev = qml.device("lightning.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def qnode(inputs, weights):
        # Encoding classical data
        qml.AngleEmbedding(inputs, wires=range(n_qubits))
        
        # Variational circuit
        qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
        
        # Measurement
        return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]

    # Define weight shapes
    n_layers = config.N_LAYERS
    weight_shapes = {"weights": (n_layers, n_qubits)}

    # Create the Quantum Layer
    qlayer = qml.qnn.TorchLayer(qnode, weight_shapes)

    # Hybrid Model: Quantum Layer -> Classical Linear Layer
    # The quantum layer outputs 'n_qubits' expectation values.
    # We map these to 'num_classes' outputs.
    class HybridModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.qlayer = qlayer
            self.fc = nn.Linear(n_qubits, num_classes)

        def forward(self, x):
            x = self.qlayer(x)
            x = self.fc(x)
            return x

    return HybridModel()
