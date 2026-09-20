import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

SEED = 42
EPOCHS = 500
LR = 0.001


# === data loading ===
def load_dataset(csv_path, target):
    df = pd.read_csv(csv_path)
    y = df[target].to_numpy().astype(float)
    X = df.drop([target], axis=1).to_numpy().astype(float)
    return X, y


# === network ===
class WineNet(nn.Module):
    def __init__(self, n_feat):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(n_feat, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.layers(x)


if __name__ == '__main__':
    X, y = load_dataset("a3_q2/a3_q2/data/wine_quality.csv", "quality")

    # 90/10 split
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.1, random_state=SEED)

    X_tr = torch.tensor(X_tr, dtype=torch.float32)
    y_tr = torch.tensor(y_tr, dtype=torch.float32).unsqueeze(1)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)

    torch.manual_seed(SEED)
    model = WineNet(X_tr.shape[1])
    loss_fn = nn.MSELoss()
    optim = torch.optim.Adam(model.parameters(), lr=LR)

    mae_list = []

    for ep in range(EPOCHS):
        # forward + backward
        pred = model(X_tr)
        loss = loss_fn(pred, y_tr)
        loss.backward()
        optim.step()
        optim.zero_grad()

        # val MAE
        with torch.no_grad():
            val_pred = model(X_val)
            mae = torch.mean(torch.abs(val_pred - y_val)).item()
        mae_list.append(mae)

        if (ep + 1) % 100 == 0:
            print(f"Epoch {ep+1}/{EPOCHS}  loss={loss.item():.4f}  val_MAE={mae:.4f}")

    # === plot ===
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, EPOCHS + 1), mae_list)
    plt.xlabel("Epoch")
    plt.ylabel("Validation MAE")
    plt.title("PyTorch Training: Validation MAE vs Epoch")
    plt.tight_layout()
    plt.savefig("workworkspace/q3_val_mae.png", dpi=150)
    plt.close()
    print(f"\nFinal val MAE: {mae_list[-1]:.4f}")
    print("Plot saved to workworkspace/q3_val_mae.png")
