import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


data = pd.read_csv("heart.csv")



target_column = 'target'

categorical_features = ["cp", "restecg", "slope", "thal", "sex", "fbs", "exang"]
numerical_features = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]

preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), numerical_features),
        ('cat', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ]), categorical_features)
    ]
)


X = preprocessor.fit_transform(data.drop(columns=target_column))
y = data[target_column].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


print(f"Training Data Shape: {X_train.shape}")
print(f"Test Data Shape: {X_test.shape}")


class FeatureNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_units: list):
        super(FeatureNetwork, self).__init__()
        layers = []
        for i in range(len(hidden_units)):
            if i == 0:
                layers.append(nn.Linear(input_dim, hidden_units[i]))
            else:
                layers.append(nn.Linear(hidden_units[i-1], hidden_units[i]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_units[-1], 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class NeuralAdditiveModel(nn.Module):
    def __init__(self, num_features: int, input_dim: int, hidden_units: list):
        super(NeuralAdditiveModel, self).__init__()
        self.feature_networks = nn.ModuleList([
            FeatureNetwork(input_dim, hidden_units) for _ in range(num_features)
        ])

    def forward(self, x):
        if x.ndim == 2:
            x = x.unsqueeze(-1)
        contributions = [network(x[:, i, :]) for i, network in enumerate(self.feature_networks)]
        output = torch.sum(torch.stack(contributions, dim=0), dim=0)
        return torch.sigmoid(output)


num_features = X_train.shape[1]
input_dim = 1
hidden_units = [32, 16]  
model = NeuralAdditiveModel(num_features, input_dim, hidden_units)
criterion = nn.BCELoss()  
optimizer = optim.Adam(model.parameters(), lr=0.001)

model.summary()


X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)


num_epochs = 100
batch_size = 32

# Training the model
for epoch in range(num_epochs):
    model.train()
    permutation = torch.randperm(X_train_tensor.size()[0])
    for i in range(0, X_train_tensor.size()[0], batch_size):
        indices = permutation[i:i+batch_size]
        batch_x, batch_y = X_train_tensor[indices], y_train_tensor[indices]

        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

    if (epoch+1) % 5 == 0:  
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

print("Training complete!")

model.eval()

with torch.no_grad():
    y_pred_prob = model(X_test_tensor)
    y_pred = (y_pred_prob > 0.5).float()
    accuracy = (y_pred.squeeze() == y_test_tensor.squeeze()).float().mean()
    print(f"Accuracy: {accuracy.item() * 100:.2f}%")

    
    y_pred_np = y_pred.numpy().squeeze()
    y_test_np = y_test_tensor.numpy().squeeze()
    precision = precision_score(y_test_np, y_pred_np)
    recall = recall_score(y_test_np, y_pred_np)
    f1 = f1_score(y_test_np, y_pred_np)
    conf_matrix = confusion_matrix(y_test_np, y_pred_np)

    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")
    print("Confusion Matrix:")
    print(conf_matrix)
