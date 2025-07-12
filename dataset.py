import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader

class ImageSharpeningDataset(Dataset):
    """
    Dataset for loading blurry and sharp image pairs from separate folders.
    Assumes that for each blurry image in `blurry_dir`, there is a
    corresponding sharp image with the same filename in `sharp_dir`.
    """
    def __init__(self, blurry_dir, sharp_dir, transform=None):
        """
        Args:
            blurry_dir (string): Directory with all the blurry images.
            sharp_dir (string): Directory with all the sharp images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.blurry_dir = blurry_dir
        self.sharp_dir = sharp_dir
        
        # Assumes the filenames in blurry_dir are the keys
        self.image_files = [f for f in os.listdir(blurry_dir) if os.path.isfile(os.path.join(blurry_dir, f))]
        
        # Define a default transform if none is provided
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize((256, 256)),
                transforms.ToTensor(), # Converts to tensor and normalizes to [0, 1]
            ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        if idx >= len(self):
            raise IndexError("Index out of range")

        # Define paths for blurry and sharp images
        img_name = self.image_files[idx]
        blurry_path = os.path.join(self.blurry_dir, img_name)
        sharp_path = os.path.join(self.sharp_dir, img_name)

        try:
            blurry_image = Image.open(blurry_path).convert('RGB')
            sharp_image = Image.open(sharp_path).convert('RGB')
        except FileNotFoundError as e:
            print(f"Error: Could not find image file {img_name} in both directories.")
            print(f"Searched paths:\n- {blurry_path}\n- {sharp_path}")
            # As a fallback, try loading the first sample
            if idx > 0:
                return self.__getitem__(0)
            else:
                raise e

        # Apply transforms
        blurry_tensor = self.transform(blurry_image)
        sharp_tensor = self.transform(sharp_image)

        return {'blurry': blurry_tensor, 'sharp': sharp_tensor, 'filename': img_name}

if __name__ == '__main__':
    # --- How to use the updated dataset loader ---
    print("Demonstrating the updated ImageSharpeningDataset...")
    
    # 1. Create dummy data directories and images
    print("Creating dummy data for demonstration...")
    os.makedirs('data/demo/blurry', exist_ok=True)
    os.makedirs('data/demo/sharp', exist_ok=True)
    
    try:
        dummy_blurry_img = Image.new('RGB', (128, 128), color='red')
        dummy_sharp_img = Image.new('RGB', (128, 128), color='green')
        
        dummy_blurry_img.save('data/demo/blurry/img1.png')
        dummy_sharp_img.save('data/demo/sharp/img1.png')
        dummy_blurry_img.save('data/demo/blurry/img2.png')
        dummy_sharp_img.save('data/demo/sharp/img2.png')

        print("Dummy data created successfully.")

        # 2. Create dataset instance
        demo_dataset = ImageSharpeningDataset(
            blurry_dir='data/demo/blurry',
            sharp_dir='data/demo/sharp'
        )
        
        # 3. Create a DataLoader
        demo_loader = DataLoader(demo_dataset, batch_size=2, shuffle=True)

        # 4. Access a sample batch
        first_batch = next(iter(demo_loader))
        blurry_batch = first_batch['blurry']
        sharp_batch = first_batch['sharp']
        filenames = first_batch['filename']

        print(f"\nNumber of samples: {len(demo_dataset)}")
        print(f"Batch size: {len(blurry_batch)}")
        print(f"Shape of blurry tensor batch: {blurry_batch.shape}") # Note the resize to 256x256
        print(f"Shape of sharp tensor batch: {sharp_batch.shape}")
        print(f"Filenames in batch: {filenames}")
        
    except Exception as e:
        print(f"An error occurred during the demonstration: {e}")
    finally:
        # Clean up dummy files
        import shutil
        if os.path.exists('data/demo'):
            shutil.rmtree('data/demo')
        print("\nCleaned up dummy data.") 