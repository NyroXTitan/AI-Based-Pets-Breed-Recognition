import os, json, torch, numpy as np
from PIL import Image, ImageOps
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    Trainer,
    TrainingArguments,
)
from evaluate import load
from sklearn.metrics import confusion_matrix, classification_report
from visualize_and_equalize_class_distribution import (
    visualize_class_distribution,
    equalize_class_distribution,
)

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

        image = np.array(image, dtype=np.uint8)  # ✅ keep original scaling

        with torch.no_grad():
            inputs = self.processor(images=image, return_tensors="pt")

        return {
            "pixel_values": inputs["pixel_values"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
        }


# ============================================================
# 🧮 Metrics
# ============================================================
def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    labels = p.label_ids

    accuracy = load("accuracy").compute(predictions=preds, references=labels)
    precision = load("precision").compute(predictions=preds, references=labels, average="weighted")
    recall = load("recall").compute(predictions=preds, references=labels, average="weighted")
    f1 = load("f1").compute(predictions=preds, references=labels, average="weighted")

    print("\n🧩 Confusion Matrix:\n", confusion_matrix(labels, preds))
    print("\n📋 Classification Report:\n", classification_report(labels, preds, zero_division=0))

    return {
        "accuracy": accuracy["accuracy"],
        "precision": precision["precision"],
        "recall": recall["recall"],
        "f1": f1["f1"],
    }


# ============================================================
# 🚀 Main Training Function
# ============================================================
def swinn():
    train_dir, val_dir = "ima/train", "ima/val"
    save_dir = "saved_model/swin"
    model_name = "microsoft/swin-base-patch4-window7-224"
    target_count = 200

    os.makedirs(save_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("🔧 Using device:", device)

    processor = AutoImageProcessor.from_pretrained(model_name, use_fast=True)

    # ============================================================
    # 📊 Load & Equalize Training Data
    # ============================================================
    raw_train = ImageFolder(train_dir, transform=transforms.Resize((224, 224)))
    raw_val = ImageFolder(val_dir, transform=transforms.Resize((224, 224)))

    raw_train = equalize_class_distribution(raw_train, target_count)

    label_map_path = os.path.join(save_dir, "trainlabel_mapping.json")
    with open(label_map_path, "w") as f:
        json.dump(raw_train.class_to_idx, f)

    num_classes = len(raw_train.class_to_idx)
    print(f"✅ Found {num_classes} classes")

    train_dataset = PetDataset(raw_train, processor)
    val_dataset = PetDataset(raw_val, processor)

    # ============================================================
    # ⚙️ Model & Training Config
    # ============================================================
    model = AutoModelForImageClassification.from_pretrained(
        model_name,
        num_labels=num_classes,
        ignore_mismatched_sizes=True,
    ).to(device)

    args = TrainingArguments(
        output_dir=f"{save_dir}/swin_output",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        fp16=True,
        num_train_epochs=10,
        learning_rate=8e-5,  # ✅ Swin works best around this
        weight_decay=0.01,  # ✅ helps generalization
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        logging_dir="./logs",
        logging_strategy="steps",
        logging_steps=1800,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_accuracy",  # ✅ added
        greater_is_better=True,  # ✅ added
        report_to="none",
        disable_tqdm=False,
        dataloader_num_workers=0,
        dataloader_pin_memory=True,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        processing_class=processor,
    )

    # ============================================================
    # 🏋️‍♂️ Train
    # ============================================================
    trainer.train()

    model_path = os.path.join(save_dir, "swin_model")
    trainer.save_model(model_path)
    processor.save_pretrained(model_path)
    print(f"✅ Model saved at: {model_path}")

    result = trainer.evaluate()
    with open(os.path.join(save_dir, "metrics_summarys.json"), "w") as f:
        json.dump(result, f, indent=4)
    print("✅ Metrics saved at:", os.path.join(save_dir, "metrics_summarys.json"))

    # ============================================================
    # 🧪 Evaluate and Save Detailed Metrics
    # ============================================================
    results = trainer.evaluate()

    # Convert numpy arrays to lists for JSON serialization
    results = {k: float(v) if isinstance(v, (np.floating, np.ndarray)) else v for k, v in results.items()}

    # Generate detailed classification report
    preds_output = trainer.predict(val_dataset)
    preds = np.argmax(preds_output.predictions, axis=1)
    labels = preds_output.label_ids

    # 🧾 Save full classification report
    class_names = list(raw_val.classes)  # ✅ get class labels from dataset
    report = classification_report(
        labels,
        preds,
        target_names=class_names,
        digits=4,
        zero_division=0,
        output_dict=True,  # get as dictionary
    )

    # Combine both HuggingFace + detailed report
    full_results = {
        "summary_metrics": results,
        "detailed_class_report": report,
    }

    # Save to JSON
    metrics_path = os.path.join(save_dir, "metrics_summary.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(full_results, f, indent=4)

    print("✅ Full metrics (summary + per-class) saved at:", metrics_path)
    torch.cuda.empty_cache()




