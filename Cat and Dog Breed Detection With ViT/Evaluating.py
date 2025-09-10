import torch
import json
from transformers import ViTForImageClassification, ViTImageProcessor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image

# ✅ Dataset class
class InferenceDataset(torch.utils.data.Dataset):
    def __init__(self, image_folder, processor):
        self.image_folder = image_folder
        self.processor = processor
        self.transform = transforms.Resize((224, 224))

    def __len__(self):
        return len(self.image_folder)

    def __getitem__(self, idx):
        image, label = self.image_folder[idx]
        if isinstance(image, torch.Tensor):
            image = transforms.ToPILImage()(image)
        if image.mode != "RGB":
            image = image.convert("RGB")
        image = self.transform(image)
        inputs = self.processor(images=image, return_tensors="pt")
        return {
            "pixel_values": inputs["pixel_values"].squeeze(0),
            "label": label
        }

# ✅ Evaluation function
def evaluate(model, dataloader, device):
    model.eval()
    preds, labels = [], []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            pixel_values = batch["pixel_values"].to(device)
            true_labels = batch["label"]

            outputs = model(pixel_values.unsqueeze(0) if len(pixel_values.shape) == 3 else pixel_values)
            logits = outputs.logits
            pred_labels = torch.argmax(logits, dim=1).cpu().numpy()

            preds.extend(pred_labels)
            labels.extend(true_labels.numpy())

    acc = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, average="weighted", zero_division=0)
    recall = recall_score(labels, preds, average="weighted", zero_division=0)
    f1 = f1_score(labels, preds, average="weighted", zero_division=0)

    print(f"\n📊 Evaluation Metrics:")
    print(f"  ✅ Accuracy:  {acc:.4f}")
    print(f"  ✅ Precision: {precision:.4f}")
    print(f"  ✅ Recall:    {recall:.4f}")
    print(f"  ✅ F1 Score:  {f1:.4f}")

# ✅ Main function
def main():
    model_path = "saved_model/vit_transformer"
    label_map_path = "saved_model/trainlabel_mapping.json"
    val_dir = "images_split/val"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("🔧 Using device:", device)

    # Load label map
    with open(label_map_path) as f:
        label_map = json.load(f)

    num_classes = len(label_map)

    # Load model & processor
    processor = ViTImageProcessor.from_pretrained(model_path)
    model = ViTForImageClassification.from_pretrained(model_path).to(device)

    # Prepare validation data
    raw_dataset = ImageFolder(val_dir)
    dataset = InferenceDataset(raw_dataset, processor)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False)

    # Evaluate
    evaluate(model, dataloader, device)

if __name__ == "__main__":
    main()
