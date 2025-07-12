import os
import requests
from tqdm import tqdm

def download_file(url, file_path):
    """Downloads a file from a URL to a given path with a progress bar."""
    response = requests.get(url, stream=True)
    response.raise_for_status()  # Raise an exception for bad status codes
    
    total_size_in_bytes = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 Kibibyte

    progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True, desc=f"Downloading {os.path.basename(file_path)}")
    
    with open(file_path, 'wb') as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
            
    progress_bar.close()
    
    if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
        print("ERROR, something went wrong during download")

def download_dncnn_weights(save_path="DnCNN_BSD68.pth"):
    """
    Downloads the pretrained DnCNN weights.
    """
    # This is the new, working URL for a compatible pretrained model.
    weights_url = "https://github.com/cszn/DnCNN/raw/master/TrainingCodes/dncnn_pytorch/net.pth"
    
    if not os.path.exists(save_path):
        print(f"Pretrained weights not found. Downloading from {weights_url}...")
        try:
            download_file(weights_url, save_path)
            print(f"Successfully downloaded weights to {save_path}")
        except Exception as e:
            print(f"Failed to download weights. Error: {e}")
            print("Please try downloading the file manually from the URL and place it in the root directory.")
    else:
        print(f"Pretrained weights already exist at {save_path}")

if __name__ == "__main__":
    download_dncnn_weights() 