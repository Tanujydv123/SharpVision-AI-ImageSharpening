import os
import shutil
from PIL import Image
from torchvision.datasets.utils import download_and_extract_archive
from tqdm import tqdm

def create_blurry_image(image, downscale_factor=4):
    """Creates a blurry version of an image by downscaling and upscaling."""
    original_size = image.size
    small_size = (original_size[0] // downscale_factor, original_size[1] // downscale_factor)
    
    # Downscale using bicubic interpolation
    downscaled = image.resize(small_size, Image.BICUBIC)
    
    # Upscale back to original size
    blurry_image = downscaled.resize(original_size, Image.BICUBIC)
    
    return blurry_image

def prepare_dataset(root_dir="data", val_split=0.1):
    """
    Downloads the BSDS500 dataset and creates blurry/sharp pairs.
    """
    dataset_url = "http://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz"
    download_root = os.path.join(root_dir, "download")
    image_dir = os.path.join(download_root, "BSR/BSDS500/data/images")

    # Download and extract if it doesn't exist
    if not os.path.exists(image_dir):
        print("Downloading BSDS500 dataset...")
        download_and_extract_archive(dataset_url, download_root, filename="BSR_bsds500.tgz")
    else:
        print("BSDS500 dataset already downloaded.")

    # Define source and destination paths
    source_images_train_dir = os.path.join(image_dir, "train")
    source_images_val_dir = os.path.join(image_dir, "val") # Using official val split
    
    train_sharp_dir = os.path.join(root_dir, "train", "sharp")
    train_blurry_dir = os.path.join(root_dir, "train", "blurry")
    val_sharp_dir = os.path.join(root_dir, "val", "sharp")
    val_blurry_dir = os.path.join(root_dir, "val", "blurry")
    
    # Create destination directories
    os.makedirs(train_sharp_dir, exist_ok=True)
    os.makedirs(train_blurry_dir, exist_ok=True)
    os.makedirs(val_sharp_dir, exist_ok=True)
    os.makedirs(val_blurry_dir, exist_ok=True)

    # Process training images
    print("Processing training images...")
    train_files = [f for f in os.listdir(source_images_train_dir) if f.endswith('.jpg')]
    for filename in tqdm(train_files):
        try:
            sharp_path = os.path.join(source_images_train_dir, filename)
            sharp_image = Image.open(sharp_path).convert("RGB")
            
            blurry_image = create_blurry_image(sharp_image)
            
            # Save both images
            sharp_image.save(os.path.join(train_sharp_dir, filename))
            blurry_image.save(os.path.join(train_blurry_dir, filename))
        except Exception as e:
            print(f"Could not process {filename}: {e}")

    # Process validation images
    print("Processing validation images...")
    val_files = [f for f in os.listdir(source_images_val_dir) if f.endswith('.jpg')]
    for filename in tqdm(val_files):
        try:
            sharp_path = os.path.join(source_images_val_dir, filename)
            sharp_image = Image.open(sharp_path).convert("RGB")

            blurry_image = create_blurry_image(sharp_image)
            
            # Save both images
            sharp_image.save(os.path.join(val_sharp_dir, filename))
            blurry_image.save(os.path.join(val_blurry_dir, filename))
        except Exception as e:
            print(f"Could not process {filename}: {e}")

    # Clean up downloaded archive
    try:
        shutil.rmtree(download_root)
        print("Cleaned up downloaded archive.")
    except OSError as e:
        print(f"Error cleaning up archive: {e}")

    print("\nDataset preparation finished!")
    print(f"Sharp training images at: {train_sharp_dir}")
    print(f"Blurry training images at: {train_blurry_dir}")
    print(f"Sharp validation images at: {val_sharp_dir}")
    print(f"Blurry validation images at: {val_blurry_dir}")

if __name__ == '__main__':
    prepare_dataset() 