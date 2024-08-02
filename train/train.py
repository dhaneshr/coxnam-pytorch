# train.py

import torch
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from lifelines.datasets import load_rossi
from lifelines.utils import concordance_index
import matplotlib.pyplot as plt

from model import CoxNAM  # Import CoxNAM from the model package
from utils.loss import cox_loss  # Import cox_loss from the utils package

# Set random seeds
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)  # If using GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def load_and_prepare_data():
    # Load the dataset
    df = load_rossi()

    # Separate features and target variables
    X = df.drop(columns=['week', 'arrest'])
    y = df['arrest']

    # Convert DataFrame to NumPy array
    X_numpy = X.to_numpy()
    y_numpy = y.to_numpy()

    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_numpy)

    # Convert to PyTorch tensors
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y_numpy, dtype=torch.float32)

    # Use real duration and event data from the dataset
    duration = df['week'].values
    event = df['arrest'].values

    # Convert to PyTorch tensors
    duration_tensor = torch.tensor(duration, dtype=torch.float32)
    event_tensor = torch.tensor(event, dtype=torch.float32)

    return X_tensor, y_tensor, duration_tensor, event_tensor, X, df

def train_model(X_tensor, duration_tensor, event_tensor, num_epochs=100, batch_size=32):
    # Define the Cox-NAM model
    num_features = X_tensor.shape[1]
    input_dim = 1
    hidden_units = [32, 16]
    coxnam_model = CoxNAM(num_features, input_dim, hidden_units)

    # Define optimizer
    optimizer = optim.Adam(coxnam_model.parameters(), lr=0.001)

    # Training loop
    for epoch in range(num_epochs):
        coxnam_model.train()
        permutation = torch.randperm(X_tensor.size()[0])
        for i in range(0, X_tensor.size()[0], batch_size):
            indices = permutation[i:i+batch_size]
            batch_x = X_tensor[indices]
            batch_duration = duration_tensor[indices]
            batch_event = event_tensor[indices]

            optimizer.zero_grad()
            risk_scores = coxnam_model(batch_x)
            loss = cox_loss(risk_scores, batch_duration, batch_event)

            # Print loss to debug
            if torch.isnan(loss):
                print(f"NaN loss detected at epoch {epoch+1}, batch {i//batch_size}")
            
            loss.backward()
            optimizer.step()

        if (epoch+1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

    # Training complete
    print("Training complete!")
    return coxnam_model

def evaluate_model(coxnam_model, X_tensor, duration, event):
    # Evaluate the model
    coxnam_model.eval()
    with torch.no_grad():
        risk_scores_test = coxnam_model(X_tensor).numpy().flatten()  # Flatten to ensure 1D array

    # Calculate C-index using lifelines
    c_index = concordance_index(duration, risk_scores_test, event)
    print(f"C-index: {c_index:.4f}")
    return c_index

def plot_shape_functions_and_distributions(model, X, feature_names):
    num_features = X.shape[1]
    fig, axes = plt.subplots(num_features, 1, figsize=(15, num_features * 6), sharex=True)
    
    for i in range(num_features):
        feature_network = model.feature_networks[i]
        feature_network.eval()
        
        # Normalize feature values for plotting
        feature_values = X[:, i]
        feature_mean = np.mean(feature_values)
        feature_std = np.std(feature_values)
        normalized_feature_values = (feature_values - feature_mean) / feature_std
        
        # Define range for plotting shape functions
        sample_inputs = np.linspace(-3, 3, 100).reshape(-1, 1)  # Standard normal range
        
        with torch.no_grad():
            sample_inputs_tensor = torch.tensor(sample_inputs, dtype=torch.float32)
            shape_values = feature_network(sample_inputs_tensor).numpy()
        
        ax = axes[i]
        
        # Plot feature distribution
        ax.hist(normalized_feature_values, bins=30, alpha=0.7, label=f'Distribution of {feature_names[i]}', color='b', density=True)
        
        # Create secondary y-axis for shape function
        ax2 = ax.twinx()
        ax2.plot(sample_inputs, shape_values, label=f'Shape Function for {feature_names[i]}', color='r')
        
        # Set labels and titles
        ax.set_xlabel('Normalized Value')
        ax.set_ylabel('Frequency')
        ax2.set_ylabel('Shape Function Output')
        ax.set_title(f'{feature_names[i]}: Distribution and Shape Function')
        
        # Add legends
        ax.legend(loc='upper left')
        ax2.legend(loc='upper right')
        
        # Grid and layout
        ax.grid(True)
    
    plt.tight_layout()
    plt.show()

def main():
    X_tensor, y_tensor, duration_tensor, event_tensor, X, df = load_and_prepare_data()
    coxnam_model = train_model(X_tensor, duration_tensor, event_tensor)
    evaluate_model(coxnam_model, X_tensor, df['week'].values, df['arrest'].values)
    feature_names = X.columns.tolist()
    plot_shape_functions_and_distributions(coxnam_model, X.to_numpy(), feature_names)

if __name__ == "__main__":
    main()
