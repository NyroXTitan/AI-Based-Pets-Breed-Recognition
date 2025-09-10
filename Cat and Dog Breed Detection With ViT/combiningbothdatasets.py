import os
import shutil

def merge_datasets(dataset1, dataset2, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # Process dataset1
    for root, dirs, files in os.walk(dataset1):
        for folder in dirs:
            src_folder = os.path.join(root, folder)
            dest_folder = os.path.join(output_dir, folder)
            os.makedirs(dest_folder, exist_ok=True)

            for file in os.listdir(src_folder):
                src_file = os.path.join(src_folder, file)
                dest_file = os.path.join(dest_folder, file)

                # If file already exists, skip (since names are unique, this shouldn’t happen)
                if not os.path.exists(dest_file):
                    shutil.copy2(src_file, dest_file)

    # Process dataset2
    for root, dirs, files in os.walk(dataset2):
        for folder in dirs:
            src_folder = os.path.join(root, folder)
            dest_folder = os.path.join(output_dir, folder)
            os.makedirs(dest_folder, exist_ok=True)

            for file in os.listdir(src_folder):
                src_file = os.path.join(src_folder, file)
                dest_file = os.path.join(dest_folder, file)

                if not os.path.exists(dest_file):
                    shutil.copy2(src_file, dest_file)


