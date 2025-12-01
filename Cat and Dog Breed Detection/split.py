import os
import shutil
import sklearn.model_selection

def split_dataset_into_train_val(source_dir, output_dir, train_ratio=0.7, val_ratio=0.1, test_ratio=0.2, seed=42):
    """
    Splits dataset of images organized by class folders into train, val, and test folders.

    Args:
        source_dir (str): Path to original dataset directory containing breed folders.
        output_dir (str): Path to output directory where 'train', 'val', and 'test' folders will be created.
        train_ratio (float): Fraction of data to reserve for training (default: 0.7 = 70%).
        val_ratio (float): Fraction of data to reserve for validation (default: 0.1 = 10%).
        test_ratio (float): Fraction of data to reserve for testing (default: 0.2 = 20%).
        seed (int): Random seed for reproducibility.
    """
    # Validate ratios sum to 1.0
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 0.001:
        raise ValueError(f"Ratios must sum to 1.0. Got: train={train_ratio}, val={val_ratio}, test={test_ratio}")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    train_dir = os.path.join(output_dir, "train")
    val_dir = os.path.join(output_dir, "val")
    test_dir = os.path.join(output_dir, "test")

    for folder in [train_dir, val_dir, test_dir]:
        if not os.path.exists(folder):
            os.makedirs(folder)

    breeds = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]

    for breed in breeds:
        breed_path = os.path.join(source_dir, breed)
        images = [f for f in os.listdir(breed_path) if f.lower().endswith((".jpg", ".jpeg", ".png"))]

        # First split: 70% train, 30% (val + test)
        train_imgs, temp_imgs = sklearn.model_selection.train_test_split(
            images,
            test_size=(val_ratio + test_ratio),  # 0.1 + 0.2 = 0.3
            random_state=seed,
            stratify=[breed]*len(images)
        )

        # Second split: From 30% temp, split into 10% val and 20% test
        # test_size = 0.2 / 0.3 = 2/3 to get 20% of total as test
        val_imgs, test_imgs = sklearn.model_selection.train_test_split(
            temp_imgs,
            test_size=test_ratio / (val_ratio + test_ratio),  # 0.2 / 0.3 = 2/3
            random_state=seed,
            stratify=[breed]*len(temp_imgs)
        )

        # Create breed subfolders in train, val, and test dirs
        breed_train_dir = os.path.join(train_dir, breed)
        breed_val_dir = os.path.join(val_dir, breed)
        breed_test_dir = os.path.join(test_dir, breed)
        os.makedirs(breed_train_dir, exist_ok=True)
        os.makedirs(breed_val_dir, exist_ok=True)
        os.makedirs(breed_test_dir, exist_ok=True)

        # Copy train images
        for img in train_imgs:
            shutil.copy2(os.path.join(breed_path, img), os.path.join(breed_train_dir, img))

        # Copy val images
        for img in val_imgs:
            shutil.copy2(os.path.join(breed_path, img), os.path.join(breed_val_dir, img))

        # Copy test images
        for img in test_imgs:
            shutil.copy2(os.path.join(breed_path, img), os.path.join(breed_test_dir, img))

    print(f"Dataset split into train ({train_ratio*100:.0f}%), val ({val_ratio*100:.0f}%), and test ({test_ratio*100:.0f}%) folders at '{output_dir}'.")
