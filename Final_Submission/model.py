# File: model.py

# --- Dependency Check ---
# This block checks for required packages and provides a clear
# error message and installation command if any are missing.
print("Verifying and installing dependencies...")
import subprocess
import sys

# A list of all pip package names we've used
package_list = [
    "opencv-python",
    "albumentations",
    "timm",
]

# Create the full pip command
# We use -q for "quiet" and --upgrade to ensure correct versions
pip_command = [sys.executable, "-m", "pip", "install", "-q", "--upgrade"] + package_list

try:
    # Run the command. check=True will raise an error if pip fails.
    subprocess.run(pip_command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print("All dependencies are successfully installed/upgraded.")
except subprocess.CalledProcessError as e:
    print("\n" + "="*80)
    print(" ERROR: Failed to install dependencies ".center(80, "="))
    print(f"The command '{' '.join(pip_command)}' failed.")
    print(f"Error details: {e}")
    print("Please try running the pip command manually in your terminal.")
    print("="*80)
    sys.exit(1) # Exit the script
except FileNotFoundError:
    print("\n" + "="*80)
    print(" ERROR: 'pip' command not found ".center(80, "="))
    print("Could not find 'pip'. Please ensure pip is installed and in your system's PATH.")
    print("="*80)
    sys.exit(1)

print("="*80 + "\n")
# --- End of Dependency Check ---
import torch
import torch.nn as nn
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
import os
from typing import Any, Iterable, List

# --- Model Architecture ---

class GpsTimmModel(nn.Module):
    def __init__(self, model_name="convnext_large.dinov3_lvd1689m", n_outputs=2):
        super(GpsTimmModel, self).__init__()
        

        self.backbone = timm.create_model(
            model_name,
            pretrained=False, # Set to False
            num_classes=0,
            global_pool=''
        )
        
        num_features = self.backbone.num_features
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.GELU(),
            nn.Dropout(p=0.2),
            nn.Linear(512, n_outputs)
        )
        
        # --- Define the Buffers ---

        self.register_buffer("gps_mean", torch.tensor([0.0, 0.0]))
        self.register_buffer("gps_std", torch.tensor([1.0, 1.0]))

    def forward(self, x):
        feature_maps = self.backbone(x)
        pooled_features = self.pooling(feature_maps)
        flattened_features = torch.flatten(pooled_features, 1)
        return self.head(flattened_features)

# --- 2. Create the Main 'Model' Class for the Evaluator ---

class Model(nn.Module):
    """
    This is the main class the evaluator will use.
    """
    def __init__(self, weights_path: str = None) -> None:
        """
        The evaluator calls this with __init__().
        The `weights_path` argument is part of the template.
        """
        super().__init__()
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # --- Build the Model ---
        self.model = GpsTimmModel(model_name="convnext_base.dinov3_lvd1689m", n_outputs=2)
        self.model.to(self.device)

        # # # --- CONVERT MODEL TO FP16 ---
        # if self.device == "cuda":
        #     self.model.half()
        # # # -----------------------------

        # --- Define Transforms ---
        self.transform = A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        
        # --- C. Set to Eval Mode ---
        self.eval()

    def eval(self) -> None:
        """Sets the model to evaluation mode."""
        self.model.eval()

    def predict(self, batch: Iterable[str]) -> List[List[float]]:
        """
        Receives a batch of image file paths and returns a list
        of [lat, lon] predictions.
        """
        images_list = []
        
        for image_path in batch:
            image = cv2.imread(image_path)
            if image is None:
                print(f"Warning: Could not read image {image_path}. Skipping.")
                continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            transformed = self.transform(image=image)['image']
            images_list.append(transformed)

        if not images_list:
            return []

        inputs = torch.stack(images_list).to(self.device)

        # if self.device == "cuda":
        #     inputs = inputs.half()

        # Run inference
        with torch.no_grad():
            scaled_preds = self.model(inputs) # Shape: [B, 2]

        # Un-scale the predictions using the loaded buffers
        unscaled_preds = (scaled_preds * self.model.gps_std) + self.model.gps_mean
        
        # Convert to a simple list of [lat, lon]
        return unscaled_preds.cpu().numpy().tolist()

def get_model() -> Model:
    """
    Factory function required by the evaluator.
    """
    return Model()