import os
import json
import random
import numpy as np
from PIL import Image, ImageOps
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, Subset
from torchvision.datasets import ImageFolder
from torchvision import transforms
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    accuracy_score,
)

from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
import evaluate
from visualize_and_equalize_class_distribution import (
    equalize_class_distribution,
    get_basic_augmentation,
)
ABC =""
# ============================================================
# 🧩 Custom Dataset
# ============================================================
class PetDataset(Dataset):
    def __init__(self, dataset, processor):
        self.dataset = dataset
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        if isinstance(image, torch.Tensor):
            image = transforms.ToPILImage()(image)

        image = ImageOps.exif_transpose(image)
        if image.mode != "RGB":
            image = image.convert("RGB")

        image = np.array(image, dtype=np.uint8)
        with torch.no_grad():
            inputs = self.processor(images=image, return_tensors="pt")

        return {
            "pixel_values": inputs["pixel_values"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
        }


# ============================================================
# 🧮 Metrics
# ============================================================

txt_num = 0

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    labels_int = labels.astype(int) if labels.dtype != int else labels

    # Metrics
    acc = accuracy_score(labels_int, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels_int, preds, average="weighted", zero_division=0
    )

    # Create classification report text
    cls_report = classification_report(labels_int, preds, zero_division=0)
    conf_mat = confusion_matrix(labels_int, preds)

    # -------------------------
    # 🔹 Save report per epoch
    # -------------------------
    global txt_num
    txt_num += 1
    global ABC
    # Build output file name

    out_dir = os.environ.get("TRAINER_SAVE_DIR", "./save")
    os.makedirs(out_dir, exist_ok=True)
    # Sanitize model name for filesystem
    safe_model_name = ABC.replace("/", "_").replace("\\", "_")
    out_path = os.path.join(out_dir, f"{safe_model_name}_epoch_{txt_num}_report.txt")

    with open(out_path, "w") as f:
        f.write(f"Epoch: {txt_num}\n")
        f.write("=" * 50 + "\n\n")
        f.write("Classification Report:\n")
        f.write(cls_report + "\n")
        f.write("\nConfusion Matrix:\n")
        f.write(str(conf_mat) + "\n\n")
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"Precision (weighted): {precision:.4f}\n")
        f.write(f"Recall (weighted): {recall:.4f}\n")
        f.write(f"F1 (weighted): {f1:.4f}\n")

    print(f"🧾 Saved detailed metrics to {out_path}")

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }



# ============================================================
# 🧪 MixUp utilities
# ============================================================
def mixup_batch(x: torch.Tensor, y: torch.Tensor, alpha: float, device=None):
    if alpha <= 0:
        return x, y, 1.0
    if device is None:
        device = x.device

    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    return mixed_x, (y, y[index], lam)


# ============================================================
# 🧰 MixUp Collator
# ============================================================

class MixupCollator:
    def __init__(self, mixup_alpha: float = 0.0, mixup_enabled: bool = False):
        self.mixup_alpha = mixup_alpha
        self.mixup_enabled = mixup_enabled

    def __call__(self, features):
        pixel_values = torch.stack([f["pixel_values"] for f in features])
        labels = torch.stack([f["labels"] for f in features])

        if self.mixup_enabled and self.mixup_alpha > 0:
            mixed_x, (y_a, y_b, lam) = mixup_batch(
                pixel_values, labels, alpha=self.mixup_alpha, device=pixel_values.device
            )
            batch = {
                "pixel_values": mixed_x,
                "y_a": y_a,
                "y_b": y_b,
                "lam": lam,
                "labels": y_a,  # eval compatibility
            }
        else:
            batch = {
                "pixel_values": pixel_values,
                "labels": labels,
            }

        return batch




# ============================================================
# ⚙️ Custom Trainer for MixUp
# ============================================================
class MixupTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        device = next(model.parameters()).device
        pixel_values = inputs.pop("pixel_values").to(device)
        labels = inputs.get("labels").to(device)

        if "y_a" in inputs and "y_b" in inputs and "lam" in inputs:
            y_a, y_b, lam = inputs.pop("y_a").to(device), inputs.pop("y_b").to(device), inputs.pop("lam")
            outputs = model(pixel_values=pixel_values, labels=None)
            logits = outputs.logits

            num_classes = logits.size(1)
            with torch.no_grad():
                t_a = torch.zeros(logits.size(0), num_classes, device=device).scatter_(1, y_a.view(-1, 1), 1.0)
                t_b = torch.zeros(logits.size(0), num_classes, device=device).scatter_(1, y_b.view(-1, 1), 1.0)
                target = lam * t_a + (1.0 - lam) * t_b

            log_probs = F.log_softmax(logits, dim=1)
            loss = - (target * log_probs).sum(dim=1).mean()
        else:
            outputs = model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss

        return (loss, outputs) if return_outputs else loss



# ============================================================
# 🚀 Main function
# ============================================================
def Model_with_freeze_unfreeze(
    train_dir,
    save_dir,
    model_name,
    target_count,
    mixup_enabled,
    default_mixup_alpha,
    num_train_epochs,
    fp16,
    metric_for_best_model,
    random_seed,
    learning_rate,
    batch_size,
    weight_decay,
    warmup_epochs,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(save_dir, exist_ok=True)
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    global ABC
    ABC = model_name
    global txt_num
    txt_num *= 0
    print(f"[INFO] model name: {ABC} {txt_num}")
    print(f"[INFO] Using device: {device}")
    print(f"[INFO] MixUp Enabled: {mixup_enabled}")

    processor = AutoImageProcessor.from_pretrained(model_name, use_fast=True)

    raw_train = ImageFolder(train_dir, transform=transforms.Resize((224, 224)))
    raw_train_eq = equalize_class_distribution(raw_train, target_count, augmentation=get_basic_augmentation())

    label_map_path = "trainlabel_mapping.json"
    with open(label_map_path, "w") as f:
        json.dump(raw_train_eq.class_to_idx, f)

    num_classes = len(raw_train_eq.class_to_idx)
    print(f"Found {num_classes} classes")

    # ============================================================
    # 📂 Prepare dataset + stratified split
    # ============================================================
    full_dataset = PetDataset(raw_train_eq, processor)
    labels_for_split = [lbl for _, lbl in raw_train_eq.samples]

    # Use StratifiedKFold to maintain class balance in split
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_seed)

    print(f"\n🔀 Performing Stratified split (for internal validation)...")
    train_idx, valid_idx = next(iter(skf.split(np.zeros(len(labels_for_split)), labels_for_split)))

    train_subset = Subset(full_dataset, train_idx)
    val_subset = Subset(full_dataset, valid_idx)
    print(f"   → Training samples: {len(train_subset)}")
    print(f"   → Validation samples: {len(val_subset)}")

    # ============================================================
    # ⚙️ Best Params (used for this single run)
    # ============================================================
    best_params = {
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "weight_decay": weight_decay,
        "mixup_alpha": default_mixup_alpha if mixup_enabled else 0.0,
    }


    print("\n💡 Using hyperparameters:")
    for k, v in best_params.items():
        print( {k}, {v})

    # ============================================================
    # 🧠 Final training with two-phase fine-tuning
    # ============================================================
    print("Training final model with best params...")
    final_model = AutoModelForImageClassification.from_pretrained(
        model_name,
        num_labels=num_classes,
        ignore_mismatched_sizes=True,
    ).to(device)

    final_collator = MixupCollator(
        mixup_alpha=best_params["mixup_alpha"],
        mixup_enabled=mixup_enabled
    )

    # -----------------------------
    # Phase 1: Warm-up (frozen backbone)
    # -----------------------------
    print(f"\n🧊 Phase 1: Training classifier head only for {warmup_epochs} epochs")

    # ✅ REPLACE the backbone freezing logic:
    def freeze_backbone(model, freeze=True):
        backbone_attrs = ["vit", "swin", "convnext", "deit", "efficientnet"]
        for name in backbone_attrs:
            if hasattr(model, name):
                backbone = getattr(model, name)
                for param in backbone.parameters():
                    param.requires_grad = not freeze
                print(f"{'Frozen' if freeze else 'Unfrozen'}: {name}")
                return True
        return False


    warmup_args = TrainingArguments(
        output_dir=os.path.join(save_dir, "final_model_warmup"),
        num_train_epochs=warmup_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        fp16=fp16,
        learning_rate=learning_rate * 2,
        weight_decay=weight_decay,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        gradient_accumulation_steps=2,
        dataloader_num_workers=0,
        save_strategy="epoch",
        eval_strategy="epoch",
        logging_strategy="epoch",
        report_to="none",
        metric_for_best_model="eval_accuracy",
        greater_is_better=True,
        load_best_model_at_end=True,
    )

    warmup_trainer = MixupTrainer(
        model=final_model,
        args=warmup_args,
        train_dataset=train_subset,
        eval_dataset=val_subset,
        data_collator=final_collator,
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=2,
                early_stopping_threshold=0.001,
            )
        ]
    )

    # Phase 1
    freeze_backbone(final_model, freeze=True)
    warmup_trainer.train()

    # -----------------------------
    # Phase 2: Unfreeze backbone
    # -----------------------------
    print("\n🔥 Phase 2: Fine-tuning entire model")


    print("\n💡 Using hyperparameters:")
    for k, v in best_params.items():
        print( {k}, {v})
    finetune_epochs = num_train_epochs - warmup_epochs
    final_args = TrainingArguments(
        output_dir=os.path.join(save_dir, "final_model"),
        num_train_epochs=finetune_epochs,
        per_device_train_batch_size=best_params["batch_size"],
        per_device_eval_batch_size=best_params["batch_size"],
        fp16=fp16,
        learning_rate=best_params["learning_rate"],
        weight_decay=best_params["weight_decay"],
        logging_dir="./logs",
        logging_strategy="steps",
        lr_scheduler_type="cosine",  # Better than linear
        warmup_ratio=0.1,
        logging_steps=1440,
        save_strategy="epoch",
        eval_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model=metric_for_best_model,
        greater_is_better=True,
        report_to="none",
        dataloader_num_workers=0,
    )

    final_trainer = MixupTrainer(
        model=final_model,
        args=final_args,
        train_dataset=train_subset,
        eval_dataset=val_subset,
        data_collator=final_collator,
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=3,
                early_stopping_threshold=0.001,
            )
        ]
    )
    # Phase 2
    freeze_backbone(final_model, freeze=False)
    final_trainer.train(resume_from_checkpoint=False)
    model_path = os.path.join(save_dir, "final_model")
    final_trainer.save_model(model_path)
    processor.save_pretrained(model_path)

    # ============================================================
    # 🧾 Save all training + tuning history
    # ============================================================

    def collect_epoch_metrics(trainer_obj):
        history = trainer_obj.state.log_history
        return [entry for entry in history if "epoch" in entry]

    warmup_logs = collect_epoch_metrics(warmup_trainer)
    finetune_logs = collect_epoch_metrics(final_trainer)
    final_eval = final_trainer.evaluate()

    print("✅ Final evaluation:", final_eval)

    meta = {
        "best_params": best_params,
        "warmup_logs": warmup_logs,
        "finetune_logs": finetune_logs,
        "final_eval": final_eval,
    }

    out_path = os.path.join(save_dir, "optuna_cv_results.json")
    with open(out_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"📊 All results saved to: {out_path}")


        # Clean up optimizer & scheduler files
    for root, _, files in os.walk(model_path):
        for f in files:
            if f in ["optimizer.pt", "scheduler.pt", "trainer_state.json", "rng_state.pth"]:
                os.remove(os.path.join(root, f))

    return meta