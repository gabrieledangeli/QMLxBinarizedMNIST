import matplotlib.pyplot as plt
import torch
import pandas as pd
from . import train_classical, train_quantum, classical_models

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    results = []

    # 1. Classical MLP
    print("\n" + "="*40)
    print("Running Classical MLP Experiment...")
    print("="*40)
    # We need to modify train_classical to return the model as well, 
    # or we can just build a fresh one to count parameters since architecture is static.
    # Let's just build one to count.
    from . import config
    mlp_model = classical_models.build_classical_model(config.INPUT_DIM, config.NUM_CLASSES, config)
    mlp_params = count_parameters(mlp_model)
    
    acc_classical = train_classical.train_model()
    results.append({
        "Model": "Classical MLP",
        "Params": mlp_params,
        "Accuracy": acc_classical
    })

    # 2. Quantum QNN
    print("\n" + "="*40)
    print("Running QNN Experiment...")
    print("="*40)
    acc_qnn, qnn_model = train_quantum.train_model(model_type="qnn")
    qnn_params = count_parameters(qnn_model)
    results.append({
        "Model": "QNN (ZZ+RealAmp)",
        "Params": qnn_params,
        "Accuracy": acc_qnn
    })

    # 3. Hybrid HQNN
    print("\n" + "="*40)
    print("Running HQNN Experiment...")
    print("="*40)
    acc_hqnn, hqnn_model = train_quantum.train_model(model_type="hqnn")
    hqnn_params = count_parameters(hqnn_model)
    results.append({
        "Model": "HQNN (QAOA+Strong)",
        "Params": hqnn_params,
        "Accuracy": acc_hqnn
    })

    # Comparison Table
    df = pd.DataFrame(results)
    print("\n" + "="*60)
    print("FINAL COMPARISON RESULTS")
    print("="*60)
    print(df.to_string(index=False))
    print("="*60)
    
    # Save to CSV
    df.to_csv("results/comparison_table.csv", index=False)

    # Plotting
    plt.figure(figsize=(10, 6))
    bars = plt.bar(df["Model"], df["Accuracy"], color=['blue', 'purple', 'green'])
    plt.ylabel('Test Accuracy')
    plt.title('Model Comparison: Classical vs QNN vs HQNN')
    plt.ylim(0, 1.05)
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                 f'{height:.4f}', ha='center', va='bottom', fontweight='bold')
        
    # Add parameter counts as text below bars
    for i, row in df.iterrows():
        plt.text(i, 0.05, f"Params:\n{row['Params']}", ha='center', color='white', fontweight='bold')

    plt.savefig('comparison_results.png')
    print("Comparison plot saved to 'comparison_results.png'")

if __name__ == "__main__":
    # Ensure results directory exists
    import os
    os.makedirs("results", exist_ok=True)
    main()
