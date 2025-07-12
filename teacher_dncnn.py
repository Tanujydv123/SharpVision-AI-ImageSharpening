import torch
import torch.nn as nn
import os

class DnCNN(nn.Module):
    """
    DnCNN model implementation. This version is modified to directly output
    a sharpened image instead of residual noise to simplify the distillation process.
    """
    def __init__(self, num_layers=17, num_features=64):
        """
        Initializes the DnCNN model.
        Args:
            num_layers (int): The number of convolutional layers.
            num_features (int): The number of features in the intermediate layers.
        """
        super(DnCNN, self).__init__()
        
        layers = []
        # First layer
        layers.append(nn.Conv2d(3, num_features, kernel_size=3, padding=1, bias=False))
        layers.append(nn.ReLU(inplace=True))
        
        # Intermediate layers
        for _ in range(num_layers - 2):
            layers.append(nn.Conv2d(num_features, num_features, kernel_size=3, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(num_features))
            layers.append(nn.ReLU(inplace=True))
        
        # Final layer
        layers.append(nn.Conv2d(num_features, 3, kernel_size=3, padding=1, bias=False))
        
        self.dncnn = nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass for the DnCNN model. The original DnCNN predicts the
        residual (noise), so we subtract it from the input to get the clean image.
        Args:
            x (torch.Tensor): Input blurry image tensor.
        Returns:
            torch.Tensor: Output sharpened image tensor.
        """
        residual = self.dncnn(x)
        return x - residual

def load_teacher_model(weights_path=None, device="cpu"):
    """
    Safely loads the teacher DnCNN model weights.
    This function can handle raw state_dict files and checkpoint files
    that contain either the model object or a 'state_dict' key.
    Args:
        weights_path (str, optional): Path to the .pth file with pretrained weights.
        device (str): The device to load the model onto ('cpu' or 'cuda').
    Returns:
        torch.nn.Module: The loaded teacher model.
    """
    print("Loading teacher model (DnCNN)...")
    # Initialize the model on the target device first.
    model = DnCNN().to(device)
    
    if weights_path:
        if not os.path.exists(weights_path):
            print(f"Error: Teacher weights file not found at '{weights_path}'")
            print("Please run 'python download_weights.py' first.")
            raise FileNotFoundError(f"[Errno 2] No such file or directory: '{weights_path}'")

        try:
            # Load the checkpoint onto the CPU first. This is safer and avoids GPU memory issues.
            checkpoint = torch.load(weights_path, map_location='cpu')
            
            state_dict = None
            # Case 1: Checkpoint is a dictionary (common modern practice).
            if isinstance(checkpoint, dict):
                if 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    # If no 'state_dict', assume the dict itself is the state_dict.
                    state_dict = checkpoint
            # Case 2: Checkpoint is the full model object (older, unsafe practice).
            elif isinstance(checkpoint, nn.Module):
                state_dict = checkpoint.state_dict()
            # Case 3: Checkpoint is the state_dict itself.
            else:
                state_dict = checkpoint
            
            if state_dict:
                # Load the extracted weights into our model instance.
                model.load_state_dict(state_dict)
                print(f"✅ Teacher model loaded successfully from '{weights_path}'.")
            else:
                raise RuntimeError("Could not extract state_dict from the checkpoint file.")

        except Exception as e:
            print(f"❌ Could not load weights for teacher model from '{weights_path}'. Error: {e}")
            print("   Please ensure the file is a valid PyTorch state_dict or checkpoint.")
            print("   Warning: Teacher model is now using random initialization.")
    else:
        print("Warning: No weights path provided for teacher. Using random initialization.")
        
    # Set the model to evaluation mode for inference.
    model.eval()
    
    return model

if __name__ == '__main__':
    # --- How to use the teacher model loader ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Load without pretrained weights (will use random init)
    teacher = load_teacher_model(device=device)
    print(f"Teacher model loaded on {device} with random weights.")
    
    # 2. To load with your own weights, provide the path
    # Create a dummy weights file for demonstration
    dummy_weights_path = "dummy_teacher_weights.pth"
    torch.save(teacher.state_dict(), dummy_weights_path)
    
    # Load with the dummy weights
    teacher_with_weights = load_teacher_model(weights_path=dummy_weights_path, device=device)
    
    # 3. Test with a dummy input
    dummy_input = torch.randn(1, 3, 256, 256).to(device)
    output = teacher_with_weights(dummy_input)
    
    print(f"\nInput shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")

    # Clean up dummy file
    if os.path.exists(dummy_weights_path):
        os.remove(dummy_weights_path)
        print(f"\nCleaned up {dummy_weights_path}.") 