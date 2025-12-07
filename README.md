# QML Challenge – Task 2: Binarized MNIST Evaluation

This repository implements **Task 2** of the ZHAW / UZH Quantum Machine Learning Challenge. It evaluates and compares a classical neural network against a hybrid quantum-classical model on the **Binarized MNIST** dataset.

## Goal

The goal is to:
1.  Implement a classical baseline (MLP).
2.  Implement a quantum/hybrid model using **PennyLane**.
3.  Train both on a subset of MNIST (e.g., digits 0 vs 1).
4.  Compare their performance in terms of **Test Accuracy**.

## Project Structure

```text
.
  src/
    config.py           # Hyperparameters and configuration
    data.py             # Data loading and preprocessing (PCA for quantum)
    classical_models.py # PyTorch MLP implementation
    quantum_models.py   # PennyLane hybrid quantum model
    train_classical.py  # Training script for classical model
    train_quantum.py    # Training script for quantum model
    compare_models.py   # Runs both and compares results
  scripts/
    download_data.py    # Helper to download data
```

## Installation

1.  **Create a virtual environment:**
    ```bash
    python -m venv .venv
    # Windows
    .\.venv\Scripts\activate
    # Mac/Linux
    source .venv/bin/activate
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

Run the experiments from the root directory of the project.

1.  **Train Classical Model:**
    ```bash
    python -m src.train_classical
    ```

2.  **Train Quantum Model:**
    ```bash
    python -m src.train_quantum
    ```

3.  **Run Comparison:**
    ```bash
    python -m src.compare_models
    ```
    This will run both training loops and generate a comparison table and a plot `comparison_results.png`.

## Models

### Classical Model
*   **Type:** Multi-Layer Perceptron (MLP)
*   **Architecture:** Input (784) -> Linear(64) -> ReLU -> Linear(32) -> ReLU -> Linear(2)
*   **Input:** Flattened 28x28 images (784 dimensions).

### Quantum Model
*   **Type:** Hybrid Quantum-Classical Classifier
*   **Preprocessing:** PCA is used to reduce the 784 dimensions to `N_QUBITS` (default: 4) features.
*   **Encoding:** `AngleEmbedding` encodes the PCA features into rotation angles.
*   **Ansatz:** `BasicEntanglerLayers` provides variational parameters.
*   **Measurement:** Expectation value of Pauli-Z on each qubit.
*   **Post-processing:** A classical linear layer maps the quantum output to class logits.

## Results

*   **Classical Accuracy:** ~99% (on 0 vs 1 task)
*   **Quantum Accuracy:** ~95-99% (depending on PCA quality and epochs)

### Observations
*   The classical model trains very quickly and achieves high accuracy on the full feature set.
*   The quantum model requires dimensionality reduction (PCA) to fit the data onto a small number of qubits (4). Despite this massive loss of information (784 -> 4 features), the quantum model performs surprisingly well, demonstrating the power of hybrid approaches on compressed data.
*   **Challenges:** Simulation time increases exponentially with qubits. Using `pennylane-lightning` helps, but keeping qubit count low is essential for rapid experimentation.

## License
MIT
