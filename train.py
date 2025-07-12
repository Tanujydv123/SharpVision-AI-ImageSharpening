import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import shutil

# --- DnCNN class definition for safe torch.load() ---
class DnCNN(nn.Module):
    def _init_(self, num_layers=17, num_features=64, channels=3):
        super(DnCNN, self)._init_()
        layers = []
        layers.append(nn.Conv2d(channels, num_features, kernel_size=3, padding=1, bias=False))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(num_layers - 2):
            layers.append(nn.Conv2d(num_features, num_features, kernel_size=3, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(num_features))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(num_features, channels, kernel_size=3, padding=1, bias=False))
        self.dncnn = nn.Sequential(*layers)

    def forward(self, x):
        residual = self.dncnn(x)
        return x - residual

# Local imports
from models import StudentCNN
from teacher_dncnn import load_teacher_model
from dataset import ImageSharpeningDataset
from utils import compute_ssim, save_visual_results

# --- Configuration ---
LEARNING_RATE = 1e-4
BATCH_SIZE = 4
NUM_EPOCHS = 50
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRAIN_BLURRY_DIR = "data/train/blurry"
TRAIN_SHARP_DIR = "data/train/sharp"
VAL_BLURRY_DIR = "data/val/blurry"
VAL_SHARP_DIR = "data/val/sharp"
TEACHER_WEIGHTS_PATH = "DnCNN_BSD68.pth"
MODEL_SAVE_PATH = "best_student_model_v2.pth"
OUTPUT_DIR = "outputs"

def setup_teacher_weights(target_path, source_path):
    if os.path.exists(target_path):
        return True
    print(f"⚠ Teacher weights '{target_path}' not found in project directory.")
    source_path_norm = os.path.normpath(source_path)
    if os.path.exists(source_path_norm):
        print(f"Found model at '{source_path_norm}'. Copying to project folder...")
        try:
            shutil.copy(source_path_norm, target_path)
            print("✅ Copy complete.")
            return True
        except Exception as e:
            print(f"❌ Failed to copy file: {e}")
            return False
    else:
        print(f"⚠ Missing teacher model. Please place the pretrained model at '{source_path_norm}'")
        print(f"or place it directly in the project folder as '{target_path}'.")
        return False

def main():
    print(f"Using device: {DEVICE}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    source_model_path = "C:/Users/Piyush/Downloads/model.pth"
    if not setup_teacher_weights(TEACHER_WEIGHTS_PATH, source_model_path):
        return

    try:
        checkpoint = torch.load(TEACHER_WEIGHTS_PATH, map_location=DEVICE, weights_only=False)
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
            teacher_model = DnCNN(channels=3)
            teacher_model.load_state_dict(state_dict)
        elif isinstance(checkpoint, dict):
            teacher_model = DnCNN(channels=3)
            teacher_model.load_state_dict(checkpoint)
        elif hasattr(checkpoint, "state_dict"):
            teacher_model = checkpoint
        else:
            raise RuntimeError("Could not extract state_dict from checkpoint.")
        teacher_model.to(DEVICE)
        teacher_model.eval()
        print("✅ Teacher model loaded.")
    except Exception as e:
        print(f"❌ Could not load teacher model: {e}")
        print("   Warning: Teacher model is now using random initialization.")
        teacher_model = DnCNN(channels=3).to(DEVICE)
        teacher_model.eval()

    print("Loading datasets...")
    if not os.path.exists(TRAIN_BLURRY_DIR):
        print("Error: Data directories not found. Please run prepare_dataset.py first.")
        return

    train_dataset = ImageSharpeningDataset(blurry_dir=TRAIN_BLURRY_DIR, sharp_dir=TRAIN_SHARP_DIR)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=False)

    val_dataset = ImageSharpeningDataset(blurry_dir=VAL_BLURRY_DIR, sharp_dir=VAL_SHARP_DIR)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=False)
    print("Datasets loaded successfully.")

    print("Initializing models...")
    student_model = StudentCNN().to(DEVICE)

    mse_loss = nn.MSELoss()

    def combined_loss(student_output, teacher_output):
        loss_mse = mse_loss(student_output, teacher_output)
        loss_ssim = 1 - compute_ssim(student_output, teacher_output)
        return 0.8 * loss_mse + 0.2 * loss_ssim

    optimizer = optim.Adam(student_model.parameters(), lr=LEARNING_RATE)
    best_val_ssim = 0.0

    print("Starting training...")
    for epoch in range(NUM_EPOCHS):
        student_model.train()
        total_train_loss = 0.0

        for batch in train_loader:
            blurry_imgs = batch['blurry'].to(DEVICE)
            with torch.no_grad():
                teacher_outputs = teacher_model(blurry_imgs)
            student_outputs = student_model(blurry_imgs)
            loss = combined_loss(student_outputs, teacher_outputs)
            total_train_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        avg_train_loss = total_train_loss / len(train_loader)

        student_model.eval()
        total_val_ssim = 0.0
        saved_visuals = False

        with torch.no_grad():
            for batch in val_loader:
                blurry_imgs = batch['blurry'].to(DEVICE)
                sharp_imgs = batch['sharp'].to(DEVICE)
                student_outputs = student_model(blurry_imgs)
                teacher_outputs = teacher_model(blurry_imgs)
                student_outputs_clamped = torch.clamp(student_outputs, 0, 1)
                ssim = compute_ssim(student_outputs_clamped, sharp_imgs)
                total_val_ssim += ssim.item()
                if not saved_visuals:
                    save_visual_results(blurry_imgs, teacher_outputs, student_outputs, sharp_imgs, epoch, OUTPUT_DIR)
                    saved_visuals = True

        avg_val_ssim = total_val_ssim / len(val_loader)
        print(f"\n--- Epoch {epoch+1}/{NUM_EPOCHS} Summary ---")
        print(f"Average Training Loss: {avg_train_loss:.4f}")
        print(f"Average Validation SSIM: {avg_val_ssim:.4f}\n")

        if avg_val_ssim > best_val_ssim:
            best_val_ssim = avg_val_ssim
            torch.save(student_model.state_dict(), MODEL_SAVE_PATH)
            print(f"✨ New best model saved with SSIM: {avg_val_ssim:.4f} to {MODEL_SAVE_PATH}")

    print("Training finished.")
    print(f"Best student model saved at {MODEL_SAVE_PATH} with SSIM: {best_val_ssim:.4f}")

if __name__ == "__main__":
    main() 