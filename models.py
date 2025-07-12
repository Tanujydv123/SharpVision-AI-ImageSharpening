import torch
import torch.nn as nn

class StudentCNN(nn.Module):
    """A lightweight CNN for image sharpening."""
    def __init__(self, channels=1):
        super(StudentCNN, self).__init__()
        self.conv1 = nn.Conv2d(channels, 64, kernel_size=9, padding=4)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=1, padding=0)
        self.conv3 = nn.Conv2d(32, channels, kernel_size=5, padding=2)
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Forward pass for the student model.
        Args:
            x (torch.Tensor): Input blurry image tensor.
        Returns:
            torch.Tensor: Output sharpened image tensor.
        """
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x

class TeacherCNN(nn.Module):
    """A deeper CNN to act as the teacher model (e.g., a simplified DnCNN)."""
    def __init__(self):
        super(TeacherCNN, self).__init__()
        layers = []
        layers.append(nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1, bias=True))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(5):
            layers.append(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=True))
            layers.append(nn.BatchNorm2d(64))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, padding=1, bias=True))
        self.dncnn = nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass for the teacher model.
        Args:
            x (torch.Tensor): Input blurry image tensor.
        Returns:
            torch.Tensor: Output sharpened image tensor from the teacher.
        """
        out = self.dncnn(x)
        return out

if __name__ == '__main__':
    # Test the models with a dummy input
    dummy_input = torch.randn(1, 1, 256, 256) # (B, C, H, W) for grayscale

    student = StudentCNN(channels=1)
    teacher = TeacherCNN()

    student_output = student(dummy_input)
    teacher_output = teacher(dummy_input)

    print(f"Student output shape: {student_output.shape}")
    print(f"Teacher output shape: {teacher_output.shape}")

    # Count parameters
    student_params = sum(p.numel() for p in student.parameters() if p.requires_grad)
    teacher_params = sum(p.numel() for p in teacher.parameters() if p.requires_grad)

    print(f"Student model has {student_params:,} trainable parameters.")
    print(f"Teacher model has {teacher_params:,} trainable parameters.") 