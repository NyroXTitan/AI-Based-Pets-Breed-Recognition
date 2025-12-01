from data_loader import extract_dataset
from fixingdatasets import  oxforddatasetrename , standforddatasetrename
from combiningbothdatasets import merge_datasets
from split import split_dataset_into_train_val


def main():
    print("Extracting Dataset")
    extract_dataset("stanford.tar", "stanford")
    extract_dataset("oxford.tar.gz", "images")

    print('\n\nFixing Dataset Folder names and images name')
    oxforddatasetrename()
    standforddatasetrename()

    print('\n\n Combining Both Datasets')
    dataset1 = r"images/images"
    dataset2 = r"stanford/Images"
    output_dir = r"merged_dataset"
    merge_datasets(dataset1, dataset2, output_dir)
    print("✅ Merge complete! Images saved in:", output_dir)

    print('\n\n Spliting Datasets For Training and Testing 70/10/20')
    split_dataset_into_train_val("ima/Images", "ima", train_ratio=0.7, val_ratio=0.1, test_ratio=0.2)

if __name__ == "__main__":
    main()

