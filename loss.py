import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import distance_transform_edt, morphological_gradient


def lovasz_grad(gt_sorted):
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / (union + 1e-6)  # ✅ FIX

    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]

    return jaccard


def lovasz_softmax_flat(probas, labels):
    if probas.numel() == 0:
        return probas * 0.

    C = probas.size(1)
    losses = []

    for c in range(C):
        fg = (labels == c).float()

        if fg.sum() == 0:
            continue

        class_pred = probas[:, c]
        errors = (fg - class_pred).abs()

        errors_sorted, perm = torch.sort(errors, descending=True)
        fg_sorted = fg[perm]

        losses.append(torch.dot(errors_sorted, lovasz_grad(fg_sorted)))

    return sum(losses) / max(len(losses), 1)


# ========================= FOCAL LOSS =========================
class FocalLoss(nn.Module):
    def __init__(self, gamma=2):
        super(FocalLoss, self).__init__()
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        # ✅ FIX: stable class weighting
        class_counts = torch.bincount(targets.flatten(), minlength=inputs.size(1)).float()
        class_weights = (class_counts.sum() / (class_counts + 1e-6))
        class_weights = class_weights / class_weights.sum()

        weights = class_weights[targets]

        return (focal_loss * weights).mean()


# ========================= LOVASZ =========================
class LovaszSoftmaxLoss(nn.Module):
    def forward(self, inputs, targets):
        probas = F.softmax(inputs, dim=1)
        probas = probas.permute(0, 2, 3, 1).contiguous().view(-1, probas.size(1))
        labels = targets.view(-1)
        return lovasz_softmax_flat(probas, labels)


# ========================= BOUNDARY LOSS =========================
class BoundaryLoss(nn.Module):
    def __init__(self, sigma=5.0):
        super(BoundaryLoss, self).__init__()
        self.sigma = sigma

    def compute_boundary_weight(self, targets):
        targets_np = targets.detach().cpu().numpy()
        weight_maps = []

        for b in range(targets_np.shape[0]):
            target = targets_np[b]

            boundary = morphological_gradient(target.astype(np.int32), size=(3, 3)) > 0

            if boundary.sum() == 0:
                weight_maps.append(np.ones_like(target, dtype=np.float32))
            else:
                dist = distance_transform_edt(~boundary)
                w = np.exp(-dist / self.sigma)
                weight_maps.append(w.astype(np.float32))

        weight_tensor = torch.from_numpy(np.stack(weight_maps)).to(targets.device)

        return weight_tensor

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        boundary_weight = self.compute_boundary_weight(targets)

        return (ce_loss * boundary_weight).mean()


# ========================= MAIN LOSS =========================
class TriComponentBoundaryAwareLoss(nn.Module):
    def __init__(self):
        super(TriComponentBoundaryAwareLoss, self).__init__()

        self.focal = FocalLoss(gamma=2)
        self.lovasz = LovaszSoftmaxLoss()
        self.boundary = BoundaryLoss(sigma=5.0)
        self.aux_bce = nn.BCEWithLogitsLoss()

    def forward(self, pred_seg, pred_aux, targets, hsv_v_variance_proxy, conditions=None):
        l_focal = self.focal(pred_seg, targets)
        l_lovasz = self.lovasz(pred_seg, targets)
        l_boundary = self.boundary(pred_seg, targets)

        # ✅ FIX: safe shape handling
        aux_target = hsv_v_variance_proxy.float()
        aux_pred = pred_aux.squeeze(1)

        if aux_pred.shape != aux_target.shape:
            aux_target = aux_target.unsqueeze(1)

        l_aux = self.aux_bce(aux_pred, aux_target)

        l_total = 0.3 * l_focal + 0.4 * l_lovasz + 0.2 * l_boundary + 0.1 * l_aux

        # Condition weighting
        if conditions is not None:
            cond_weights = torch.tensor(
                [1.0, 1.2, 1.3, 1.4, 1.5],
                device=l_total.device,
                dtype=torch.float32
            )
            weights_mean = cond_weights[conditions].mean()
            l_total = l_total * weights_mean

        return l_total


# ========================= OHEM =========================
class BoundaryOHEM:
    def __init__(self, crop_size=128, margin=3, threshold=0.7):
        self.crop_size = crop_size
        self.margin = margin
        self.threshold = threshold

    def mine_crops(self, images, pred_seg, targets, conditions=None):
        probas = F.softmax(pred_seg, dim=1)
        pred_class = probas.argmax(dim=1)
        pred_max_prob, _ = probas.max(dim=1)

        wrong_preds = (pred_class != targets) & (pred_max_prob > self.threshold)

        B, H, W = targets.shape
        targets_np = targets.detach().cpu().numpy()

        crop_images = []
        crop_targets = []
        crop_conditions = []

        for b in range(B):
            target = targets_np[b]

            boundary = morphological_gradient(target.astype(np.int32), size=(3, 3)) > 0

            if boundary.sum() > 0:
                dist = distance_transform_edt(~boundary)
                near_boundary = torch.tensor(dist <= self.margin, device=targets.device)
            else:
                near_boundary = torch.zeros((H, W), dtype=torch.bool, device=targets.device)

            hard_pixels = wrong_preds[b] & near_boundary

            if hard_pixels.sum() > 0:
                y_coords, x_coords = torch.where(hard_pixels)
                idx = torch.randint(0, len(y_coords), (1,)).item()

                cy, cx = y_coords[idx].item(), x_coords[idx].item()

                y1 = max(0, cy - self.crop_size // 2)
                x1 = max(0, cx - self.crop_size // 2)
                y2 = min(H, y1 + self.crop_size)
                x2 = min(W, x1 + self.crop_size)

                if y2 - y1 < self.crop_size:
                    y1 = max(0, y2 - self.crop_size)
                if x2 - x1 < self.crop_size:
                    x1 = max(0, x2 - self.crop_size)

                crop_images.append(images[b:b+1, :, y1:y2, x1:x2])
                crop_targets.append(targets[b:b+1, y1:y2, x1:x2])

                if conditions is not None:
                    crop_conditions.append(conditions[b:b+1])

        if len(crop_images) > 0:
            if conditions is not None:
                return (
                    torch.cat(crop_images, dim=0),
                    torch.cat(crop_targets, dim=0),
                    torch.cat(crop_conditions, dim=0)
                )
            else:
                return (
                    torch.cat(crop_images, dim=0),
                    torch.cat(crop_targets, dim=0),
                    None
                )

        return None, None, None