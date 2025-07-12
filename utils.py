import torch
import torch.nn.functional as F
from math import exp
import os
from torchvision.utils import save_image, make_grid

def gaussian(window_size, sigma):
    """Generates a gaussian kernel."""
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    """Creates a gaussian window for SSIM calculation."""
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def compute_ssim(img1, img2, window_size=11, val_range=1.0):
    """
    Computes the Structural Similarity Index (SSIM) between two images.
    Args:
        img1 (torch.Tensor): The first image tensor (B, C, H, W).
        img2 (torch.Tensor): The second image tensor (B, C, H, W).
        window_size (int): The size of the gaussian window.
        val_range (float): The value range of the images (e.g., 1.0 for [0, 1]).
    Returns:
        torch.Tensor: The SSIM value.
    """
    if img1.size() != img2.size():
        raise ValueError("Input images must have the same dimensions.")

    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size//2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size//2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size//2, groups=channel) - mu1_mu2

    C1 = (0.01 * val_range) ** 2
    C2 = (0.03 * val_range) ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

def save_visual_results(blurry, teacher, student, sharp, epoch, output_dir="outputs"):
    """
    Saves a grid of images comparing blurry, teacher, student, and sharp images.
    
    Args:
        blurry (torch.Tensor): The blurry input tensor.
        teacher (torch.Tensor): The teacher's output tensor.
        student (torch.Tensor): The student's output tensor.
        sharp (torch.Tensor): The ground truth sharp tensor.
        epoch (int): The current epoch number.
        output_dir (str): The directory to save the output images.
    """
    # Ensure the output directory for the current epoch exists
    epoch_dir = os.path.join(output_dir, f"epoch_{epoch+1}")
    os.makedirs(epoch_dir, exist_ok=True)
    
    # Take the first image from the batch for visualization
    blurry = blurry[0].cpu()
    teacher = teacher[0].cpu().clamp(0, 1)
    student = student[0].cpu().clamp(0, 1)
    sharp = sharp[0].cpu()
    
    # Create a grid of images
    grid = make_grid([blurry, teacher, student, sharp], nrow=4, padding=2)
    
    # Save the grid
    # We use a unique name for each validation batch, e.g., based on the first image filename
    # For simplicity, we just save the first batch's comparison
    save_path = os.path.join(epoch_dir, "comparison_batch_0.png")
    save_image(grid, save_path)

if __name__ == '__main__':
    # Test the SSIM function
    img1 = torch.rand(1, 3, 256, 256)
    img2 = torch.rand(1, 3, 256, 256)
    
    # Perfect match
    ssim_val_identical = compute_ssim(img1, img1)
    print(f"SSIM between identical images: {ssim_val_identical.item():.4f}")

    # Random images
    ssim_val_random = compute_ssim(img1, img2)
    print(f"SSIM between random images: {ssim_val_random.item():.4f}")

    # Test the save_visual_results function
    print("\nTesting save_visual_results function...")
    dummy_output_dir = "test_outputs"
    if not os.path.exists(dummy_output_dir):
        os.makedirs(dummy_output_dir)

    # Create dummy tensors (as if from a batch of size 4)
    blurry_t = torch.rand(4, 3, 256, 256)
    teacher_t = torch.rand(4, 3, 256, 256)
    student_t = torch.rand(4, 3, 256, 256)
    sharp_t = torch.rand(4, 3, 256, 256)

    save_visual_results(blurry_t, teacher_t, student_t, sharp_t, epoch=0, output_dir=dummy_output_dir)
    
    # Verify file was created
    expected_file = os.path.join(dummy_output_dir, "epoch_1", "comparison_batch_0.png")
    if os.path.exists(expected_file):
        print(f"Successfully created visual comparison at: {expected_file}")
    else:
        print(f"Failed to create visual comparison image.")

    # Clean up
    import shutil
    shutil.rmtree(dummy_output_dir)
    print("Cleaned up test output directory.") 