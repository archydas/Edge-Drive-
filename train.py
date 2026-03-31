import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader
import torch.nn.functional as F
import csv
import numpy as np
from scipy.ndimage import morphological_gradient

from model import EdgeDriveModel
from loss import TriComponentBoundaryAwareLoss


# ================= METRICS =================
def compute_metrics(pred, target):
    pred = pred.astype(np.int32)
    target = target.astype(np.int32)

    tp = np.sum((pred == 1) & (target == 1))
    fp = np.sum((pred == 1) & (target == 0))
    fn = np.sum((pred == 0) & (target == 1))

    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)

    iou = tp / (tp + fp + fn + 1e-6)

    # Boundary F1
    pred_boundary = morphological_gradient(pred, size=(3, 3)) > 0
    target_boundary = morphological_gradient(target, size=(3, 3)) > 0

    tp_b = np.sum(pred_boundary & target_boundary)
    fp_b = np.sum(pred_boundary) - tp_b
    fn_b = np.sum(target_boundary) - tp_b

    bf1 = (2 * tp_b) / (2 * tp_b + fp_b + fn_b + 1e-6)

    return precision, recall, f1, iou, bf1


# ================= TRAIN =================
def train(model, train_loader, val_loader, epochs=30, device='cuda'):
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10)

    # balanced classes
    class_weights = torch.tensor([0.4, 0.6]).to(device)

    criterion = TriComponentBoundaryAwareLoss().to(device)

    best_iou = 0

    with open("metrics.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch", "train_loss", "val_loss",
            "accuracy", "miou", "bf1",
            "precision", "recall", "f1"
        ])

    for epoch in range(epochs):
        # ================= TRAIN =================
        model.train()
        total_loss = 0

        for images, targets, hsv_v_proxy, conditions in train_loader:
            images = images.to(device)
            targets = targets.to(device)
            hsv_v_proxy = hsv_v_proxy.to(device)
            conditions = conditions.to(device)

            optimizer.zero_grad()

            pred_seg, pred_aux, cond_out = model(images, condition=conditions)

            loss_seg = criterion(pred_seg, pred_aux, targets, hsv_v_proxy, conditions)

            ce_loss = F.cross_entropy(pred_seg, targets, weight=class_weights)

            loss = loss_seg + 0.5 * ce_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()

        train_loss = total_loss / len(train_loader)

        # ================= VALIDATION =================
        model.eval()

        val_loss = 0
        metrics_list = []

        with torch.no_grad():
            for v_images, v_targets, v_proxy, v_conditions in val_loader:
                v_images = v_images.to(device)
                v_targets = v_targets.to(device)

                outputs = model(v_images, condition=v_conditions)

                if len(outputs) == 3:
                    v_seg, v_aux, _ = outputs
                else:
                    v_seg, _ = outputs
                    v_aux = torch.zeros_like(v_seg[:, :1, :, :])

                loss = criterion(v_seg, v_aux, v_targets, v_proxy, v_conditions)
                val_loss += loss.item()

                preds = v_seg.argmax(dim=1).cpu().numpy()
                targets_np = v_targets.cpu().numpy()

                for i in range(preds.shape[0]):
                    metrics = compute_metrics(preds[i], targets_np[i])
                    metrics_list.append(metrics)

        val_loss /= len(val_loader)

        precision, recall, f1, iou, bf1 = np.mean(metrics_list, axis=0)

        # REAL accuracy (pixel-wise)
        acc = np.mean([
            np.mean(preds[i] == targets_np[i])
            for i in range(len(preds))
        ])

        print(f"\nEpoch {epoch+1}/{epochs}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        print(f"Acc: {acc:.4f} | mIoU: {iou:.4f} | BF1: {bf1:.4f}")
        print(f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")

        # save best
        if iou > best_iou:
            best_iou = iou
            torch.save(model.state_dict(), "best_model.pth")
            print("🔥 Best model saved!")

        with open("metrics.csv", "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch+1, train_loss, val_loss,
                acc, iou, bf1,
                precision, recall, f1
            ])

    return model


# ================= MAIN =================
if __name__ == "__main__":
    from dataset import NuScenesDrivableDataset
    from torch.utils.data import random_split

    dataroot = r"C:\Users\hp\Downloads\Edge Drive new\Edge Drive usecase\Edge Drive usecase"

    dataset = NuScenesDrivableDataset(dataroot)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    model = EdgeDriveModel()

    model = train(model, train_loader, val_loader, epochs=3, device=device)

    torch.save(model.state_dict(), "final_model.pth")

    print("Training complete!")