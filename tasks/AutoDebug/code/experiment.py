"""
AutoDebug: A lightweight debug task for testing the InternAgent pipeline.
Uses synthetic regression data with a simple MLP model.
Training completes in ~5 seconds on GPU, ~10 seconds on CPU.
"""

import os
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


class SimpleMLP(nn.Module):
    """Simple 2-layer MLP for regression"""
    def __init__(self, input_dim=20, hidden_dims=[64, 32], dropout=0.0):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x).squeeze(-1)


def generate_data(n_samples=1000, n_features=20, noise=10.0, random_state=42):
    """Generate synthetic regression data"""
    X, y = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        noise=noise,
        random_state=random_state
    )
    return X.astype(np.float32), y.astype(np.float32)


def train_model(model, train_loader, val_loader, epochs=100, lr=0.001, device='cpu'):
    """Train the model and return training history"""
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(epochs):
        # Training
        model.train()
        train_losses = []
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                pred = model(X_batch)
                loss = criterion(pred, y_batch)
                val_losses.append(loss.item())

        history['train_loss'].append(np.mean(train_losses))
        history['val_loss'].append(np.mean(val_losses))

    return model, history


def evaluate_model(model, X_test, y_test, device='cpu'):
    """Evaluate model and return metrics"""
    model.eval()
    X_tensor = torch.FloatTensor(X_test).to(device)

    with torch.no_grad():
        predictions = model(X_tensor).cpu().numpy()

    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)

    return {
        'mse': float(mse),
        'r2': float(r2),
        'mae': float(mae)
    }


def run_experiment(config=None):
    """
    Run the experiment with given configuration.

    Args:
        config: dict with optional keys:
            - hidden_dims: list of hidden layer dimensions (default: [64, 32])
            - dropout: dropout rate (default: 0.0)
            - epochs: number of training epochs (default: 100)
            - lr: learning rate (default: 0.001)
            - batch_size: batch size (default: 32)

    Returns:
        dict with metrics and training info
    """
    # Default configuration
    if config is None:
        config = {}

    hidden_dims = config.get('hidden_dims', [64, 32])
    dropout = config.get('dropout', 0.0)
    epochs = config.get('epochs', 100)
    lr = config.get('lr', 0.001)
    batch_size = config.get('batch_size', 32)

    # Device setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Generate data
    print("Generating synthetic data...")
    X, y = generate_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )

    # Create data loaders
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Create and train model
    print(f"Training MLP with hidden_dims={hidden_dims}, dropout={dropout}")
    model = SimpleMLP(input_dim=20, hidden_dims=hidden_dims, dropout=dropout)

    start_time = time.time()
    model, history = train_model(
        model, train_loader, val_loader,
        epochs=epochs, lr=lr, device=device
    )
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")

    # Evaluate
    metrics = evaluate_model(model, X_test, y_test, device=device)
    print(f"Test MSE: {metrics['mse']:.4f}")
    print(f"Test R2: {metrics['r2']:.4f}")
    print(f"Test MAE: {metrics['mae']:.4f}")

    return {
        'metrics': metrics,
        'config': {
            'hidden_dims': hidden_dims,
            'dropout': dropout,
            'epochs': epochs,
            'lr': lr,
            'batch_size': batch_size
        },
        'training_time': training_time,
        'history': {
            'final_train_loss': history['train_loss'][-1],
            'final_val_loss': history['val_loss'][-1]
        }
    }


def main():
    """Main entry point for experiment"""
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Run experiment with default config (baseline)
    results = run_experiment()

    # Save results
    output = {
        'mse': results['metrics']['mse'],
        'r2': results['metrics']['r2'],
        'mae': results['metrics']['mae'],
        'training_time': results['training_time'],
        'config': results['config']
    }

    # Save to final_info.json in parent directory (run_X folder)
    parent_dir = os.path.dirname(script_dir)
    output_path = os.path.join(parent_dir, 'final_info.json')

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    return output


if __name__ == '__main__':
    main()
