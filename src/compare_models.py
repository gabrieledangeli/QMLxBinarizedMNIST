import matplotlib.pyplot as plt
import torch
import pandas as pd
import os
from . import train_classical, train_quantum, classical_models

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def plot_history(histories):
    """
    Plots training and validation loss for all models.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    
    colors = {
        'Classical MLP': '#3498db', # Blue
        'QNN (ZZ+RealAmp)': '#9b59b6', # Purple
        'HQNN (QAOA+Strong)': '#2ecc71' # Green
    }
    
    for i, (model_name, history) in enumerate(histories.items()):
        ax = axes[i]
        epochs = range(1, len(history['train_loss']) + 1)
        
        ax.plot(epochs, history['train_loss'], 'o-', label='Train Loss', color=colors[model_name], alpha=0.7)
        ax.plot(epochs, history['val_loss'], 's--', label='Val Loss', color=colors[model_name], linestyle='--')
        
        ax.set_title(f"{model_name}", fontsize=14)
        ax.set_xlabel("Epochs")
        if i == 0:
            ax.set_ylabel("Loss")
        ax.legend()
        ax.grid(True, linestyle=':', alpha=0.6)
        
    plt.tight_layout()
    plt.savefig('results/loss_comparison.png', dpi=300)
    print("Loss comparison plot saved to 'results/loss_comparison.png'")

def main():
    results = []
    histories = {}

    # 1. Classical MLP
    print("\n" + "="*40)
    print("Running Classical MLP Experiment...")
    print("="*40)
    from . import config
    mlp_model = classical_models.build_classical_model(config.INPUT_DIM, config.NUM_CLASSES, config)
    mlp_params = count_parameters(mlp_model)
    
    acc_classical, hist_classical = train_classical.train_model()
    results.append({
        "Model": "Classical MLP",
        "Params": mlp_params,
        "Accuracy": acc_classical
    })
    histories["Classical MLP"] = hist_classical

    # 2. Quantum QNN
    print("\n" + "="*40)
    print("Running QNN Experiment...")
    print("="*40)
    acc_qnn, qnn_model, hist_qnn = train_quantum.train_model(model_type="qnn")
    qnn_params = count_parameters(qnn_model)
    results.append({
        "Model": "QNN (ZZ+RealAmp)",
        "Params": qnn_params,
        "Accuracy": acc_qnn
    })
    histories["QNN (ZZ+RealAmp)"] = hist_qnn

    # 3. Hybrid HQNN
    print("\n" + "="*40)
    print("Running HQNN Experiment...")
    print("="*40)
    acc_hqnn, hqnn_model, hist_hqnn = train_quantum.train_model(model_type="hqnn")
    hqnn_params = count_parameters(hqnn_model)
    results.append({
        "Model": "HQNN (QAOA+Strong)",
        "Params": hqnn_params,
        "Accuracy": acc_hqnn
    })
    histories["HQNN (QAOA+Strong)"] = hist_hqnn

    # Comparison Table
    df = pd.DataFrame(results)
    print("\n" + "="*60)
    print("FINAL COMPARISON RESULTS")
    print("="*60)
    print(df.to_string(index=False))
    print("="*60)
    
    # Save to CSV
    os.makedirs("results", exist_ok=True)
    df.to_csv("results/comparison_table.csv", index=False)

    # Plotting Accuracy Bar Chart
    plt.figure(figsize=(10, 6))
    
    # Elegant colors
    bar_colors = ['#3498db', '#9b59b6', '#2ecc71'] # Blue, Purple, Green
    
    bars = plt.bar(df["Model"], df["Accuracy"], color=bar_colors, width=0.6)
    plt.ylabel('Test Accuracy', fontsize=12)
    plt.title('Model Comparison: Classical vs QNN vs HQNN', fontsize=14)
    plt.ylim(0, 1.05)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                 f'{height:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=11)
        
    # Add parameter counts as text below bars
    for i, row in df.iterrows():
        plt.text(i, 0.05, f"Params:\n{row['Params']}", ha='center', color='white', fontweight='bold', fontsize=10)

    plt.tight_layout()
    plt.savefig('results/comparison_results.png', dpi=300)
    print("Comparison plot saved to 'results/comparison_results.png'")

    # Plot Loss Curves
    plot_history(histories)

if __name__ == "__main__":
    main()
