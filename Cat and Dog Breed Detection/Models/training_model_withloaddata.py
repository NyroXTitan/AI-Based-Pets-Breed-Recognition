
import os
import json
import random
import numpy as np
from PIL import Image, ImageOps
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from torchvision import transforms
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
from visualize_and_equalize_class_distribution import (
    equalize_class_distribution,
    get_basic_augmentation,
    MixupCutmixCollator,
    mixup_criterion,
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
# ⚙ Custom Trainer for MixUp/CutMix
# ============================================================
class AugmentedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        device = next(model.parameters()).device
        pixel_values = inputs.pop("pixel_values").to(device)
        labels = inputs.get("labels").to(device)

        if "y_a" in inputs and "y_b" in inputs and "lam" in inputs:
            y_a = inputs.pop("y_a").to(device)
            y_b = inputs.pop("y_b").to(device)
            lam = inputs.pop("lam")
            outputs = model(pixel_values=pixel_values, labels=None)
            logits = outputs.logits
            num_classes = logits.size(1)
            loss = mixup_criterion(logits, y_a, y_b, lam, num_classes,device = device)
        else:
            outputs = model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss

        return (loss, outputs) if return_outputs else loss



# ============================================================
# 🚀 Main function
# ============================================================
def Model_with_freeze_unfreeze(
    train_dir,
    val_dir,
    save_dir,
    model_name,
    target_count,
    mix_augment_enabled,
    mixup_alpha,
    cutmix_alpha,
    num_train_epochs,
    fp16,
    metric_for_best_model,
    random_seed,
    learning_rate,
    batch_size,
    weight_decay,
    warmup_epochs,
    p_mixup=0.3,
    p_cutmix=0.7,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(save_dir, exist_ok=True)
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    print(f"[INFO] model name: {model_name}")
    print(f"[INFO] Using device: {device}")


    processor = AutoImageProcessor.from_pretrained(model_name, use_fast=True)

    raw_train = ImageFolder(train_dir, transform=transforms.Resize((224, 224)))
    raw_train_eq = equalize_class_distribution(raw_train, target_count, augmentation=get_basic_augmentation())

    label_map_path = "trainlabel_mapping.json"
    with open(label_map_path, "w") as f:
        json.dump(raw_train_eq.class_to_idx, f)

    num_classes = len(raw_train_eq.class_to_idx)
    print(f"Found {num_classes} classes")

    # ============================================================
    # 📂 Prepare datasets
    # ============================================================
    train_dataset = PetDataset(raw_train, processor)
    
    # Load validation dataset from external directory
    if not os.path.exists(val_dir):
        raise ValueError(f"Validation directory not found: {val_dir}")
    raw_val = ImageFolder(val_dir, transform=transforms.Resize((224, 224)))
    
    # Verify class mappings match between train and val
    if raw_val.class_to_idx != raw_train.class_to_idx:
        missing_in_train = set(raw_val.class_to_idx.keys()) - set(raw_train.class_to_idx.keys())
        missing_in_val = set(raw_train.class_to_idx.keys()) - set(raw_val.class_to_idx.keys())
        error_msg = "Validation and training datasets have mismatched classes:\n"
        if missing_in_train:
            error_msg += f"  Classes in validation but not in training: {missing_in_train}\n"
        if missing_in_val:
            error_msg += f"  Classes in training but not in validation: {missing_in_val}"
        raise ValueError(error_msg)
    
    val_dataset = PetDataset(raw_val, processor)
    
    print(f"\n📂 Using external validation directory: {val_dir}")
    print(f"   → Training samples: {len(train_dataset)}")
    print(f"   → Validation samples: {len(val_dataset)}")

    # ============================================================
    # ⚙ Best Params (used for this single run)
    # ============================================================
    best_params = {
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "weight_decay": weight_decay,
        "mixup_alpha": mixup_alpha if mix_augment_enabled else 0.0,
        "cutmix_alpha": cutmix_alpha if mix_augment_enabled else 0.0,
        "p_mixup": p_mixup if mix_augment_enabled else 0.0,
        "p_cutmix": p_cutmix if mix_augment_enabled else 0.0,
    }


    print("\n💡 Using hyperparameters:")
    for k, v in best_params.items():
        print(f"{k}: {v}")

    # ============================================================
    # 🧠 Final training with two-phase fine-tuning
    # ============================================================
    print("Training final model with best params...")
    final_model = AutoModelForImageClassification.from_pretrained(
        model_name,
        num_labels=num_classes,
        ignore_mismatched_sizes=True,
        trust_remote_code=True,
    ).to(device)

    print("Model type:", type(final_model))
    # Print top-level attributes for debugging
    print("Top-level attributes:", [a for a in dir(final_model) if not a.startswith("_")][:80])

    final_collator = MixupCutmixCollator(
        mixup_alpha=best_params["mixup_alpha"],
        cutmix_alpha=best_params["cutmix_alpha"],
        p_mixup=best_params["p_mixup"],
        p_cutmix=best_params["p_cutmix"],
        enabled=mix_augment_enabled,
        device=device,
    )


    # -----------------------------
    # Phase 1: Warm-up (frozen backbone)
    # -----------------------------
    print(f"\n🧊 Phase 1: Training classifier head only for {warmup_epochs} epochs")

    def freeze_backbone(model, freeze=True):
        """
        Robust backbone freeze/unfreeze helper.

        Strategy:
        1. Try known HF backbone attributes (vit, swin, convnext, deit, efficientnet, resnet).
        2. Try common timm-wrapper attribute names (model, timm_model, backbone, base_model, model_vision).
        3. If still not found, scan named_modules() for a classifier-like module and unfreeze only it when freeze=True.
        4. If nothing matches, fallback to freezing/unfreezing all params (with an explanatory message).
        """
        backbone_attrs = ["vit", "swin", "convnext", "deit", "efficientnet", "resnet"]
        for name in backbone_attrs:
            if hasattr(model, name):
                backbone = getattr(model, name)
                for param in backbone.parameters():
                    param.requires_grad = not freeze
                print(f"{'Frozen' if freeze else 'Unfrozen'} backbone: {name} (HF attribute)")
                return True

        # Common wrapper attribute names that may hold the timm model
        candidate_attrs = ["model", "timm_model", "backbone", "base_model", "model_vision", "body", "encoder"]
        for attr in candidate_attrs:
            if hasattr(model, attr):
                candidate = getattr(model, attr)
                # If candidate exposes get_classifier(), treat it as timm-like
                try:
                    if hasattr(candidate, "get_classifier") or hasattr(candidate, "classifier") or hasattr(candidate,
                                                                                                           "head") or hasattr(
                            candidate, "fc"):
                        if freeze:
                            # freeze all first
                            for p in model.parameters():
                                p.requires_grad = False
                            # unfreeze the classifier only
                            try:
                                # Prefer using get_classifier() if available
                                if hasattr(candidate, "get_classifier"):
                                    classifier = candidate.get_classifier()
                                elif hasattr(candidate, "classifier"):
                                    classifier = candidate.classifier
                                elif hasattr(candidate, "head"):
                                    classifier = candidate.head
                                elif hasattr(candidate, "fc"):
                                    classifier = candidate.fc
                                else:
                                    classifier = None

                                if classifier is not None:
                                    for p in classifier.parameters():
                                        p.requires_grad = True
                                    print(f"Unfrozen classifier head found on attribute '{attr}'")
                                    return True
                                else:
                                    # Could not locate classifier object, fallback to unfreeze all
                                    for p in model.parameters():
                                        p.requires_grad = True
                                    print(
                                        f"⚠ Could not extract classifier module from attribute '{attr}'. Unfrozen all params.")
                                    return True
                            except Exception as e:
                                print(f"⚠ Error while unfreezing classifier at attr '{attr}': {e}. Unfreezing all.")
                                for p in model.parameters():
                                    p.requires_grad = True
                                return True
                        else:
                            # unfreeze everything for phase 2
                            for p in model.parameters():
                                p.requires_grad = True
                            print(f"Unfrozen all layers (via '{attr}')")
                            return True
                except Exception as e:
                    print(f"⚠ Checking attr '{attr}' raised: {e}")

        # Last-resort: scan modules and try to detect a classifier module by heuristics
        classifier_candidates = []
        for name, module in model.named_modules():
            if name == "":
                continue
            # Heuristic checks
            if hasattr(module, "get_classifier"):  # timm style
                classifier_candidates.append((name, module))
            elif any(hasattr(module, a) for a in ("classifier", "head", "fc")):
                classifier_candidates.append((name, module))

        if classifier_candidates:
            # Pick the most-final looking candidate (the longest name usually)
            classifier_candidates.sort(key=lambda x: len(x[0]), reverse=True)
            cls_name, cls_module = classifier_candidates[0]
            print(f"Detected classifier candidate: '{cls_name}' -> {type(cls_module)}")
            if freeze:
                # freeze all then unfreeze classifier
                for p in model.parameters():
                    p.requires_grad = False
                for p in cls_module.parameters():
                    p.requires_grad = True
                print(f"Unfrozen classifier module '{cls_name}' (heuristic)")
            else:
                for p in model.parameters():
                    p.requires_grad = True
                print("Unfrozen all layers (heuristic path)")
            return True

        # Fallback: could not find specific backbone/classifier. Inform and return False
        print("⚠ Could not find a known backbone/classifier to selectively freeze/unfreeze.")
        print(f"   Model type: {type(model)}")
        print("   As a safe fallback, freezing/unfreezing all model params.")
        for p in model.parameters():
            p.requires_grad = not freeze
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

    warmup_trainer = AugmentedTrainer(
        model=final_model,
        args=warmup_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
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
        print(f"{k}: {v}")
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

    final_trainer = AugmentedTrainer(
        model=final_model,
        args=final_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
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
