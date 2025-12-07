import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import argparse
from . import config, data, quantum_models

def train_model(model_type="qnn"):
    # Determine if we need PCA based on model type
    # QNN uses PCA features (n_qubits)
    # HQNN uses full features (784) and learns embedding
    use_pca = (model_type == "qnn")
    
    print(f"Loading data for {model_type.upper()} (PCA={use_pca})...")
    x_train, y_train, x_test, y_test = data.load_binarized_mnist(
        n_qubits=config.N_QUBITS, 
        use_pca=use_pca
    )
    
    # Create DataLoaders
    train_dataset = TensorDataset(x_train, y_train)
    test_dataset = TensorDataset(x_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    # Build Model
    print(f"Building {model_type.upper()} model...")
    model = quantum_models.build_quantum_model(model_type, config.N_QUBITS, config.NUM_CLASSES, config)
    
    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.QUANTUM_LR)
    
    # Scheduler: ReduceLROnPlateau
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

    print(f"Starting {model_type.upper()} Training...")
    for epoch in range(config.EPOCHS_QUANTUM):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.EPOCHS_QUANTUM}")
        for inputs, labels in pbar:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            pbar.set_postfix({'loss': running_loss/total, 'acc': correct/total})

        # Validation (using test set for simplicity here, ideally split val)
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_acc = val_correct / val_total
        print(f"Epoch {epoch+1} - Test Acc: {val_acc:.4f}")
        
        # Step scheduler
        scheduler.step(val_acc)

    # Final Evaluation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print(f"{model_type.upper()} Model Final Test Accuracy: {accuracy:.4f}")
    return accuracy, model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default="qnn", choices=["qnn", "hqnn"], help="Type of quantum model to train")
    args = parser.parse_args()
    
    train_model(model_type=args.model_type)
