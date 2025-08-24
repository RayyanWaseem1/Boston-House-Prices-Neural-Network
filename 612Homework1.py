import numpy as np
import pandas as pd
from time import perf_counter
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, RegressorMixin
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import warnings

warnings.filterwarnings("ignore")

# Step1: Load data
print("Step1: Importing libraries and loading data …")
COLS = [
    "CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE",
    "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT", "MEDV"
]
df = pd.read_csv("/Users/rayyanwaseem/Desktop/612/hw1/housing.csv", header = None, names = COLS)
X = df.drop("MEDV", axis=1).values.astype(np.float32)
y = df["MEDV"].values.astype(np.float32)
print(f" → Dataset shape : {df.shape}\n")

# Step2: Split data
print("Step2: Splitting into X and y …")
print(f" → X shape: {X.shape}   y shape: {y.shape}\n")

# Step3: PyTorch Regressor Wrapper
class TorchRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, input_dim=13, hidden_layers=(64, 32), epochs=100, batch_size=32, lr=0.001):
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr

    def _build_model(self):
        layers = []
        in_dim = self.input_dim
        for h in self.hidden_layers:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        return nn.Sequential(*layers)

    def fit(self, X, y):
        self.model = self._build_model()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()

        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y.reshape(-1, 1), dtype=torch.float32)
        loader = DataLoader(TensorDataset(X_tensor, y_tensor), batch_size=self.batch_size, shuffle=True)

        self.model.train()
        for _ in range(self.epochs):
            for xb, yb in loader:
                optimizer.zero_grad()
                loss = loss_fn(self.model(xb), yb)
                loss.backward()
                optimizer.step()
        return self

    def predict(self, X):
        self.model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            return self.model(X_tensor).squeeze().numpy()

# Step4: Cross-validation
def evaluate_model(hidden_layers):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    scores = []

    for train_idx, test_idx in kf.split(X_scaled):
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = TorchRegressor(hidden_layers=hidden_layers, epochs=100)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = ((y_pred - y_test) ** 2).mean()
        scores.append(mse)

    return np.mean(scores), np.std(scores)

# Step5: Baseline model
print("Step3: Building a two-hidden-layer MLP (64,32) …")
print(" → Model: TorchRegressor(hidden_layers=(64, 32), epochs=100, batch_size=32)\n")

print("Step4: Standardisation handled automatically inside each CV fold.\n")

print("Step5: 10-fold cross-validation for (64,32) model → ", end="")
mean_mse, std_mse = evaluate_model((64, 32))
print(f"MSE = {mean_mse:6.2f}  ± {std_mse:.2f}\n")

# Step6: Single hidden layer sweep
print("Step6: Single-layer sweep (units → MSE):")
units_list = [4, 8, 16, 32, 64, 128]
single_results = {}
for units in units_list:
    mse, _ = evaluate_model((units,))
    single_results[units] = mse
    print(f"  {units:>6} units : MSE {mse:6.2f}")
best_units = min(single_results, key=single_results.get)
print(f" → Best width = {best_units} units (MSE {single_results[best_units]:.2f})")

# Step7: Depth experiment
print("Step7: Depth experiment with 32 neurons/layer:")
for depth in [1, 2, 3, 4, 5]:
    start = perf_counter()
    mse, _ = evaluate_model((32,) * depth)
    elapsed = perf_counter() - start
    print(f"  {depth:>2} layer(s) : MSE {mse:6.2f}   (fit ≈{elapsed:5.2f}s)")
