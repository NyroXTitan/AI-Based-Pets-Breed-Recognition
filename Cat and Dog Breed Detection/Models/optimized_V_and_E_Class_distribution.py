import matplotlib.pyplot as plt
from collections import Counter
from torchvision import transforms
import random
from torch.utils.data import Dataset
from tqdm import tqdm
import torch
import numpy as np
from PIL import Image

import albumentations as A


# ---------------------------
# EqualizedDataset wrapper
# ---------------------------
class EqualizedDataset(Dataset):
    def __init__(self, samples, class_to_idx):
        """
        samples: list of (PIL.Image, label)
        """
        self.samples = samples
        self.class_to_idx = class_to_idx

    def __getitem__(self, idx):
        return self.samples[idx]

    def __len__(self):
        return len(self.samples)


# ---------------------------
# Basic augmentations
# ---------------------------
def get_basic_augmentation(size=(224, 224)):
    """
    Uses Albumentations for much faster augmentation.
    """
    return A.Compose([
        A.RandomResizedCrop(size=size, scale=(0.8, 1.0)),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=20, p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.8),
    ])



# ============================================================
# 🧰 MixUp + CutMix Collator
# ============================================================
class MixupCutmixCollator:
    def __init__(
        self,
        mixup_alpha: float = 0.0,
        cutmix_alpha: float = 0.0,
        p_mixup: float = 0.0,
        p_cutmix: float = 0.0,
        enabled: bool = False,
        device: torch.device | None = 'cpu',
    ):
        """
        MixUp + CutMix combo collator.

        - If enabled=False -> no mixing, plain batch.
        - If both alphas <= 0 -> no mixing.
        - With probabilities p_mixup / p_cutmix it chooses which augmentation to apply.
        - Uses mixup_images_labels / cutmix_images_labels from optimized_V_and_E file.
        """
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.p_mixup = p_mixup
        self.p_cutmix = p_cutmix
        self.enabled = enabled
        self.device = device  # can be set from training code

    def __call__(self, features):
        pixel_values = torch.stack([f["pixel_values"] for f in features])
        labels = torch.stack([f["labels"] for f in features])

        # No mixing conditions
        if (not self.enabled) or (self.mixup_alpha <= 0 and self.cutmix_alpha <= 0):
            return {
                "pixel_values": pixel_values,
                "labels": labels,
            }

        r = random.random()

        # Decide which augmentation to apply
        if r < self.p_mixup and self.mixup_alpha > 0:

            mixed_x, y_a, y_b, lam = mixup_images_labels(
                pixel_values, labels, alpha=self.mixup_alpha
            )
        elif r < self.p_mixup + self.p_cutmix and self.cutmix_alpha > 0:

            mixed_x, y_a, y_b, lam = cutmix_images_labels(
                pixel_values, labels, alpha=self.cutmix_alpha
            )
        else:
            # No mixing for this batch
            return {
                "pixel_values": pixel_values,
                "labels": labels,
            }

        return {
            "pixel_values": mixed_x,
            "y_a": y_a,
            "y_b": y_b,
            "lam": lam,
            "labels": y_a,  # for eval compatibility in compute_metrics
        }

# ---------------------------
# MixUp & CutMix helpers
# ---------------------------
def mixup_images_labels(x, y, alpha=0.4, device: torch.device | None = 'cpu',):
    """
    x: tensor (B, C, H, W)
    y: tensor (B,) long labels
    returns mixed_x, mixed_y (soft labels as float tensor (B, num_classes))
    NOTE: requires num_classes to be provided externally when converting labels to one-hot.
    This helper returns (mixed_x, (y_a, y_b, lam)) to allow computing loss outside,
    or you can call mixup_criterion which I also provide below.
    """
    if alpha <= 0:
        return x, y, y, 1.0

    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def rand_bbox(size, lam):
    # size: (B, C, H, W)
    W = size[3]
    H = size[2]
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def cutmix_images_labels(x, y, alpha=1.0, device='cpu'):
    """
    Returns mixed images and (y_a, y_b, lam)
    """
    if alpha <= 0:
        return x, y, y, 1.0

    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device)

    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bby1:bby2, bbx1:bbx2] = x[index, :, bby1:bby2, bbx1:bbx2]
    # adjust lambda to actual area ratio
    lam_adj = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size(-1) * x.size(-2)))
    y_a, y_b = y, y[index]
    return x, y_a, y_b, lam_adj


# ---------------------------
# Loss helpers for soft labels (MixUp)
# ---------------------------
import torch.nn.functional as F

def mixup_criterion(logits, y_a, y_b, lam, num_classes, device='cuda'):
    """
    Compute loss for mixup with soft labels.
    logits: (B, C)
    y_a, y_b: (B,) long labels
    lam: scalar
    returns: scalar loss
    """
    # produce soft targets
    with torch.no_grad():
        target = torch.zeros((logits.size(0), num_classes), device=device)
        target.scatter_(1, y_a.view(-1,1), lam)
        target.scatter_(1, y_b.view(-1,1), 1 - lam)
    log_probs = F.log_softmax(logits, dim=1)
    loss = - (target * log_probs).sum(dim=1).mean()
    return loss


# ---------------------------
# Equalize / oversample dataset ---
# ---------------------------
def equalize_class_distribution(dataset, target_count, augmentation=None):
    """
    dataset: torchvision.datasets.ImageFolder-like where dataset[i] returns (PIL.Image, label)
    target_count: desired number of samples per class after equalization
    augmentation: Albumentations transform (PIL -> NumPy)
    Returns: EqualizedDataset object; new_samples are tuples (PIL.Image, label)
    """
    label_to_indices = {}
    for idx, (_, label) in enumerate(dataset):
        label_to_indices.setdefault(label, []).append(idx)

    class_to_idx = dataset.class_to_idx
    new_samples = []

    if augmentation is None:
        augmentation = get_basic_augmentation()

    print("🔁 equalizing class distribution (using optimized Albumentations)...")
    for label, indices in tqdm(label_to_indices.items(), desc="Equalizing Classes", total=len(label_to_indices)):
        samples = [dataset[i] for i in indices]  # (PIL, label)

        # Convert all original samples to NumPy arrays ONCE for augmentation
        # We'll convert back to PIL for storage to match your PetDataset
        pil_samples = [s[0] for s in samples]
        labels = [s[1] for s in samples]

        if len(samples) >= target_count:
            # take deterministic subset
            new_samples.extend(samples[:target_count])
        else:
            # keep originals
            new_samples.extend(samples)

            # create augmented examples
            num_to_add = target_count - len(samples)
            for i in range(num_to_add):
                # Choose a random original image (as PIL)
                img_pil, lbl = random.choice(samples)

                # 1. Convert PIL to NumPy
                img_np = np.array(img_pil)

                # 2. Apply FAST augmentation on NumPy array
                aug_data = augmentation(image=img_np)
                aug_img_np = aug_data['image']

                # 3. Convert back to PIL to store
                # Your PetDataset expects PIL, so we convert back
                aug_img_pil = Image.fromarray(aug_img_np)

                new_samples.append((aug_img_pil, lbl))

    return EqualizedDataset(new_samples, class_to_idx)


# ---------------------------
# visualization
# ---------------------------
def visualize_class_distribution(dataset, title="Class Distribution", class_names=None):
    """
    dataset: either torchvision ImageFolder, or EqualizedDataset-like where items are (img, label)
    """
    label_counts = Counter([label for _, label in dataset])
    labels, counts = zip(*sorted(label_counts.items(), key=lambda x: x[0]))
    if class_names is None and hasattr(dataset, 'classes'):
        class_names = [dataset.classes[i] for i in labels]
    elif class_names is not None:
        class_names = [class_names[i] for i in labels]
    else:
        class_names = [str(i) for i in labels]

    plt.figure(figsize=(24, 10))
    plt.bar(class_names, counts)
    plt.title(title, fontsize=16)
    plt.xlabel("Classes", fontsize=14)
    plt.ylabel("Number of Images", fontsize=14)
    plt.xticks(rotation=90, fontsize=8)
    plt.tight_layout()
    plt.show()


