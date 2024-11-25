import os
import json

from PIL import Image
from tqdm import tqdm


def create_data_lists(train_folders, test_folders, min_size, output_folder):
    print("\nCreating data lists... this may take some time.\n")

    # Training images
    train_images = list()
    print("Processing training folders...")
    for d in tqdm(train_folders, desc="Train Folders"):
        for i in tqdm(os.listdir(d), desc=f"Processing {d}", leave=False):
            img_path = os.path.join(d, i)
            try:
                img = Image.open(img_path, mode="r")
                if img.width >= min_size and img.height >= min_size:
                    # Filter the images
                    train_images.append(img_path)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")

    print(f"\nThere are {len(train_images)} images in the training set.")
    with open(os.path.join(output_folder, "train_images.json"), "w") as j:
        json.dump(train_images, j)

    # Test images
    print("Processing test folders...")
    for d in tqdm(test_folders, desc="Test Folders"):
        test_images = list()
        test_name = os.path.basename(d)
        for i in tqdm(os.listdir(d), desc=f"Processing {d}", leave=False):
            img_path = os.path.join(d, i)
            try:
                img = Image.open(img_path, mode="r")
                if img.width >= min_size and img.height >= min_size:
                    test_images.append(img_path)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")

        print(f"\nThere are {len(test_images)} images in the {test_name} test data.\n")

        with open(
            os.path.join(output_folder, test_name + "_test_images.json"), "w"
        ) as j:
            json.dump(test_images, j)

    print(f"JSON files with image paths created and saved to {output_folder}.")


if __name__ == "__main__":
    train_folder = ["/root/autodl-tmp/train2014", "/root/autodl-tmp/val2014"]
    test_folder = [
        "/root/autodl-tmp/BSDS100",
        "/root/autodl-tmp/Set5",
        "/root/autodl-tmp/Set14",
    ]

    min_size = 100
    output_folder = "./data"
    create_data_lists(train_folder, test_folder, min_size, output_folder)

    print("Data lists created!")
