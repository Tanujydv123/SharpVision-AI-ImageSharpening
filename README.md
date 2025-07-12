# Image Sharpening using Knowledge Distillation

This project demonstrates how to use knowledge distillation to train a lightweight student model for image sharpening, guided by a more complex teacher model.

⚠️ Important : First, extract the ZIP file after downloading, then navigate to the extracted folder and run the code as instructed.

## Project Structure

```
.
├── models.py               # Contains Teacher and Student CNN architectures
├── dataset.py              # PyTorch Dataset for loading blurry/sharp image pairs
├── utils.py                # Utility functions, including SSIM computation
├── train.py                # Main training script for knowledge distillation
├── requirements.txt        # Python dependencies
└── best_student_model.pth  # Output path for the best saved model
```

## How It Works

1.  **Teacher Model**: A pre-trained, deep CNN (`TeacherCNN` in `models.py`) that excels at image sharpening.
2.  **Student Model**: A smaller, faster CNN (`StudentCNN` in `models.py`) that we want to train.
3.  **Knowledge Distillation**: The student model is not trained on the ground truth sharp images directly. Instead, it is trained to mimic the *output* of the teacher model.
    -   A blurry image is fed to both the teacher and the student.
    -   The loss (Mean Squared Error) is calculated between the student's output and the teacher's output.
    -   This loss is used to update the student's weights.
4.  **Evaluation**: Although the student learns from the teacher, its performance is evaluated against the actual ground truth sharp images using the Structural Similarity Index (SSIM) to see how well it's learning the sharpening task.

## How to Run

### 1. Installation

First, install the required Python packages:

```bash
pip install -r requirements.txt
```

### 2. Prepare the Data

The `train.py` script will create dummy data to run out-of-the-box. However, for actual training, you must structure your data as follows:

```
data/
├── train/
│   ├── 001/
│   │   ├── blurry.png
│   │   └── sharp.png
│   ├── 002/
│   │   ├── blurry.png
│   │   └── sharp.png
│   └── ...
└── val/
    ├── 101/
    │   ├── blurry.png
    │   └── sharp.png
    └── ...
```

-   Create a `data` directory with `train` and `val` subdirectories.
-   Inside `train` and `val`, create a separate folder for each pair of images.
-   Each folder must contain a `blurry.png` (the input) and a `sharp.png` (the ground truth).

### 3. Start Training

Once your data is in place, run the training script:

```bash
python train.py
```

The script will:
- Use your GPU if `cuda` is available.
- Load the datasets.
- Train the student model by learning from the teacher.
- Evaluate the student on the validation set after each epoch.
- Save the student model with the best validation SSIM to `best_student_model.pth`.

## Customization

-   **Models**: You can modify the architectures of the `StudentCNN` and `TeacherCNN` in `models.py`. For a real use case, you would load pre-trained weights into the `TeacherCNN`.
-   **Loss Function**: The knowledge distillation loss is set to `nn.MSELoss()` in `train.py`. You could experiment with other losses like `nn.L1Loss()` or even a perceptual loss.
-   **Hyperparameters**: Adjust the `LEARNING_RATE`, `BATCH_SIZE`, and `NUM_EPOCHS` at the top of `train.py` to suit your dataset.
