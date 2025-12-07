# QML Challenge – Task 2: Binarized MNIST Evaluation

This repository implements **Task 2** of the ZHAW / UZH Quantum Machine Learning Challenge. It evaluates and compares classical and quantum machine learning approaches on the **Binarized MNIST** dataset.

The implementation is inspired by the paper **"Hybrid quantum neural networks show strongly reduced need for free parameters in entity matching" (Scientific Reports, 2025)**, adapting the architectures to the Binarized MNIST classification task (digits 0 vs 1).

## Goal

The goal is to:
1.  Implement a **Classical Baseline (MLP)**.
2.  Implement a **Quantum Neural Network (QNN)** with minimal parameters.
3.  Implement a **Hybrid Quantum Neural Network (HQNN)** with trainable classical embedding.
4.  Compare their performance in terms of **Test Accuracy** and **Number of Parameters**.

## Project Structure

```text
.
  src/
    config.py           # Hyperparameters and configuration
    data.py             # Data loading (PennyLane datasets) and preprocessing
    classical_models.py # PyTorch MLP implementation
    quantum_models.py   # PennyLane QNN and HQNN implementations
    train_classical.py  # Training script for classical model
    train_quantum.py    # Training script for quantum models (QNN/HQNN)
    compare_models.py   # Runs all experiments and generates comparison plots
  scripts/
    download_data.py    # Helper to download data
  results/              # Output folder for plots and tables
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

### 1. Run Full Comparison (Recommended)
This script trains all three models (MLP, QNN, HQNN), saves the results, and generates comparison plots.
```bash
python -m src.compare_models
```
**Outputs in `results/`:**
*   `comparison_table.csv`: Table with accuracy and parameter counts.
*   `comparison_results.png`: Bar chart comparing test accuracy.
*   `loss_comparison.png`: Training and validation loss curves.

### 2. Train Individual Models
You can also train models individually:

*   **Classical MLP:**
    ```bash
    python -m src.train_classical
    ```

*   **Quantum Neural Network (QNN):**
    ```bash
    python -m src.train_quantum --model_type qnn
    ```

*   **Hybrid Quantum Neural Network (HQNN):**
    ```bash
    python -m src.train_quantum --model_type hqnn
    ```

## Models Implemented

### 1. Classical MLP (Baseline)
*   **Architecture:** Input (784) -> Linear(64) -> ReLU -> Linear(32) -> ReLU -> Linear(2)
*   **Parameters:** ~52k
*   **Input:** Flattened 28x28 images.

### 2. Quantum Neural Network (QNN)
*   **Inspiration:** "Quantum neural network (QNN)" section of the paper.
*   **Preprocessing:** PCA reduces input to `N_QUBITS` (4) features.
*   **Encoding:** `ZZFeatureMap`-like encoding (Hadamard + RZ + ZZ interactions).
*   **Ansatz:** `RealAmplitudes`-like ansatz with linear entanglement (RY rotations + CNOT chain).
*   **Parameters:** Very few (e.g., ~42 parameters for 4 qubits, 2 layers).
*   **Goal:** Demonstrate learning capability with minimal parameters.

### 3. Hybrid Quantum Neural Network (HQNN)
*   **Inspiration:** "Hybrid quantum neural network (HQNN)" section of the paper.
*   **Architecture:** Trainable Classical Embedding -> Quantum Circuit -> Linear Output.
*   **Embedding:** Maps 784 inputs to `N_QUBITS` features (learned end-to-end).
*   **Quantum Circuit:** `QAOAEmbedding` + `StronglyEntanglingLayers`.
*   **Parameters:** ~50k (dominated by the classical embedding).
*   **Goal:** Combine classical feature extraction with quantum processing.

## Results & Observations

*   **Classical MLP:** Achieves high accuracy (~99.8%) with fast training.
*   **QNN:** Achieves surprisingly high accuracy (~99.5%) despite having **3 orders of magnitude fewer parameters** (tens vs thousands). This confirms the paper's finding regarding the parameter efficiency of quantum models.
*   **HQNN:** Performs well (~98%), leveraging the trainable embedding to adapt the high-dimensional input to the quantum circuit.

## License
MIT
