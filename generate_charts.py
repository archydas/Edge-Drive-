import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
import pandas as pd
import torch
import time
from sklearn.metrics import confusion_matrix
from model import EdgeDriveModel

print("Generating Hackathon presentation charts...")

sns.set_theme(style="darkgrid")

# ================= LOAD METRICS =================
if not os.path.exists("metrics.csv"):
    raise FileNotFoundError("metrics.csv not found.")

df = pd.read_csv("metrics.csv")

epochs = df["epoch"].values
train_loss = df["train_loss"].values
val_loss = df["val_loss"].values
miou = df["miou"].values
bf1 = df["bf1"].values
precision = df["precision"].values
recall = df["recall"].values
f1 = df["f1"].values

# ================= OUTPUT DIR =================
os.makedirs("outputs", exist_ok=True)

# ================= 1. LOSS =================
plt.figure(figsize=(10,6))
plt.plot(epochs, train_loss, label='Train Loss', linewidth=2.5)
plt.plot(epochs, val_loss, label='Validation Loss', linewidth=2.5)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training & Validation Loss')
plt.legend()
plt.tight_layout()
plt.savefig('outputs/loss_curve.png', dpi=300)
plt.close()

# ================= 2. MIOU =================
plt.figure(figsize=(10,6))
plt.plot(epochs, miou, linewidth=3)
plt.axhline(y=miou[-1], linestyle='--', label=f'Final {miou[-1]:.2f}')
plt.ylim(0, 1)
plt.xlabel('Epochs')
plt.ylabel('mIoU')
plt.title('mIoU Progression')
plt.legend()
plt.tight_layout()
plt.savefig('outputs/miou_curve.png', dpi=300)
plt.close()

# ================= 3. BF1 =================
plt.figure(figsize=(10,6))
plt.plot(epochs, bf1, linewidth=3)
plt.axhline(y=bf1[-1], linestyle='--', label=f'Final {bf1[-1]:.2f}')
plt.ylim(0, 1)
plt.xlabel('Epochs')
plt.ylabel('Boundary F1')
plt.title('Boundary F1 Score')
plt.legend()
plt.tight_layout()
plt.savefig('outputs/bf1_curve.png', dpi=300)
plt.close()

# ================= 4. PRECISION / RECALL =================
plt.figure(figsize=(10,6))
plt.plot(epochs, precision, label='Precision')
plt.plot(epochs, recall, label='Recall')
plt.plot(epochs, f1, label='F1 Score')
plt.xlabel('Epochs')
plt.ylabel('Score')
plt.title('Precision / Recall / F1')
plt.legend()
plt.tight_layout()
plt.savefig('outputs/prf_curve.png', dpi=300)
plt.close()

# ================= 5. CONFUSION MATRIX =================
if os.path.exists("preds.npy") and os.path.exists("targets.npy"):
    preds = np.load("preds.npy")
    targets = np.load("targets.npy")

    cm = confusion_matrix(targets.flatten(), preds.flatten())

    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Ground Truth')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('outputs/confusion_matrix.png', dpi=300)
    plt.close()

# ================= 6. REAL FPS =================
def measure_fps():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = EdgeDriveModel().to(device)

    if os.path.exists("best_model.pth"):
        model.load_state_dict(torch.load("best_model.pth", map_location=device))

    model.eval()

    dummy = torch.randn(1, 3, 256, 512).to(device)

    # warmup
    for _ in range(10):
        _ = model(dummy)

    start = time.time()
    runs = 50

    for _ in range(runs):
        _ = model(dummy)

    end = time.time()

    fps = runs / (end - start)
    return fps

fps = measure_fps()

plt.figure(figsize=(6,5))
plt.bar(["EdgeDrive"], [fps])
plt.ylabel("FPS")
plt.title(f"Inference Speed ({fps:.1f} FPS)")
plt.tight_layout()
plt.savefig("outputs/fps.png", dpi=300)
plt.close()

# ================= 7. FINAL METRICS BAR =================
labels = ["Accuracy", "mIoU", "BF1", "Precision", "Recall", "F1"]
values = [
    df["accuracy"].values[-1],
    miou[-1],
    bf1[-1],
    precision[-1],
    recall[-1],
    f1[-1]
]

plt.figure(figsize=(10,6))
plt.bar(labels, values)
plt.ylim(0,1)
plt.title("Final Model Performance")
plt.tight_layout()
plt.savefig("outputs/final_metrics.png", dpi=300)
plt.close()

print("All charts generated successfully (REAL data, no hardcoding).")