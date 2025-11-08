import torch
from training_model_withloaddata import Model_with_freeze_unfreeze

# ============================================================
# 🚀 Helper to run a single model
# ============================================================
def run_model(tag, cfg):
    print(f"\n==============================")
    print(f"🚀 Now running {tag}")
    print(f"==============================")

    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True

    out = Model_with_freeze_unfreeze(
        train_dir="ima/train",
        save_dir=cfg["save_dir"],
        model_name=cfg["model_name"],
        target_count=200,
        mixup_enabled=True,
        learning_rate=cfg["learning_rate"],
        batch_size=cfg["batch_size"],
        weight_decay=cfg["weight_decay"],
        default_mixup_alpha=cfg["mixup_alpha"],
        num_train_epochs =cfg["num_train_epochs"],
        warmup_epochs=cfg["warmup_epochs"],
        fp16=True,
        metric_for_best_model="eval_accuracy",
        random_seed=42,
    )

    print(f"✅ {tag} done. Summary: {out}\n")
    return out


# ============================================================
# ⚙️ Best Known Hyperparameters (from tuning)
# ============================================================
best_configs = {
    "ViT": {
        "model_name": "google/vit-base-patch16-224-in21k",
        "save_dir": "saved_model/ViT",
        "learning_rate": 5e-5,  # ⬇️ More conservative
        "batch_size": 8,  # Smaller for ViT
        "weight_decay": 0.06,  # ⬆️ Stronger regularization
        "mixup_alpha": 0.25,  # More conservative
        "num_train_epochs": 12,
        "warmup_epochs": 4,
    },
    "EfficientNet": {
        "model_name": "google/efficientnet-b0",
        "save_dir": "saved_model/EfficientNet",
        "learning_rate": 1e-4,      # ⬇️ Reduced from 3e-4
        "batch_size": 16,           # ⬆️ Add explicit batch size
        "weight_decay": 0.01,       # ⬆️ Increased regularization
        "mixup_alpha": 0.2,         # ⬇️ Reduced
        "num_train_epochs": 15,     # More epochs needed
        "warmup_epochs": 4,         # Explicit warmup
    },
    "Swin": {
        "model_name": "microsoft/swin-base-patch4-window7-224",
        "save_dir": "saved_model/Swin",
        "learning_rate": 1.5e-5,
        "batch_size": 12,
        "weight_decay": 0.05,
        "mixup_alpha": 0.25,
        "num_train_epochs": 12,
        "warmup_epochs": 4,
    },
    "ConvNeXt": {
        "model_name": "facebook/convnext-base-224-22k",
        "save_dir": "saved_model/ConvNeXt",
        "learning_rate": 5e-5,
        "batch_size": 8,
        "weight_decay": 0.05,
        "mixup_alpha": 0.2,
        "num_train_epochs": 12,
        "warmup_epochs": 4,
    },
}


# ============================================================
# 🏁 Run All Models
# ============================================================
if __name__ == "__main__":
    results = {}
    for tag, cfg in best_configs.items():
        results[tag] = run_model(tag, cfg)

    print("\n==============================")
    print("🏆 All models finished. Summary:")
    print("==============================")
    for tag, metrics in results.items():
        print(f"{tag}: {metrics}")
