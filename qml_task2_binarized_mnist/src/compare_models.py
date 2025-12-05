import matplotlib.pyplot as plt
from . import train_classical, train_quantum

def main():
    print("Running Classical Model Experiment...")
    acc_classical = train_classical.train_model()
    
    print("\nRunning Quantum Model Experiment...")
    acc_quantum = train_quantum.train_model()
    
    print("\n" + "="*40)
    print("COMPARISON RESULTS")
    print("="*40)
    print(f"{'Model':<20} | {'Test Accuracy':<15}")
    print("-" * 40)
    print(f"{'Classical MLP':<20} | {acc_classical:.4f}")
    print(f"{'Quantum Hybrid':<20} | {acc_quantum:.4f}")
    print("="*40)

    # Plotting
    models = ['Classical MLP', 'Quantum Hybrid']
    accuracies = [acc_classical, acc_quantum]

    plt.figure(figsize=(8, 6))
    plt.bar(models, accuracies, color=['blue', 'purple'])
    plt.ylabel('Test Accuracy')
    plt.title('Classical vs Quantum Model Accuracy on Binarized MNIST')
    plt.ylim(0, 1.0)
    for i, v in enumerate(accuracies):
        plt.text(i, v + 0.01, f"{v:.4f}", ha='center', fontweight='bold')
    
    plt.savefig('comparison_results.png')
    print("Comparison plot saved to 'comparison_results.png'")

if __name__ == "__main__":
    main()
