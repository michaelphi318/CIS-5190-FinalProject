# File: 4_validate_pytorch.py

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from transformers import AutoImageProcessor, AutoModel,Dinov2Model,ViTModel,Swinv2Model,ConvNextV2Model
import pandas as pd
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import joblib
import cv2
import os
import argparse  # We'll use this to tell the script which model to test
from tqdm import tqdm
import haversine as hs

# --- 1. Configuration ---
# Your directory structure
VAL_DIR = r"D:\UPenn\CIS-5190\project\data\validation"
SCALER_PATH = r"D:\UPenn\CIS-5190\project\models\gps_scaler.joblib"# The scaler saved by your training script

# Model settings (must match training)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_HEIGHT = 224
IMG_WIDTH = 224

# --- 2. Model Definitions ---
# We must re-define the exact model architectures from your notebook
# so PyTorch knows how to load the saved weights.

def build_efficientnet_v2s():
    """Builds the EfficientNet-v2s model structure."""
    model = models.efficientnet_v2_s(weights=None) # No pre-trained weights needed here
    num_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.5, inplace=True),
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.Dropout(p=0.5, inplace=True),
        nn.Linear(512, 2)
    )
    return model

def build_resnet50():
    """Builds the ResNet-50 model structure."""
    model = models.resnet50(weights=None)
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, 2)
    )
    return model

class GpsDinoModel(nn.Module):
    """
    A wrapper class that combines the DINOv2 base model with a custom
    regression head for our GPS prediction task.
    """
    def __init__(self, base_model, num_features):
        super(GpsDinoModel, self).__init__()
        self.dino = base_model
        # Define the regression head (must match training script)
        self.head = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(num_features, 512),
            nn.GELU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, 2) # Output: 2 neurons
        )
            
    def forward(self, x):
        outputs = self.dino(x)
        # Use the [CLS] token (must match training script)
        cls_token = outputs.last_hidden_state[:, 0]
        return self.head(cls_token)

def build_dinov2_model():
    """Builds the DINOv2 model structure using Hugging Face transformers."""
    model_name = "facebook/dinov2-base"
    # We load the base model here, but the weights will be loaded later
    base_model = Dinov2Model.from_pretrained(model_name)
    num_features = base_model.config.hidden_size
    
    # Create our full model
    model = GpsDinoModel(base_model, num_features)
    return model

# Copied from your notebook to match the fusion model structure
class RegressionHead(nn.Module):
    def __init__(self, in_features, n_outputs=2):
        super(RegressionHead, self).__init__()
        self.fc1 = nn.Linear(in_features, 512)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, n_outputs)

    def forward(self, x):
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def build_convexnet_model():
    model = models.convnext_base(weights='DEFAULT')
    for param in model.features.parameters():
        param.requires_grad = False
    
    num_features = model.classifier[2].in_features
    original_layernorm = model.classifier[0]
    model.classifier = nn.Sequential(
        original_layernorm,
        nn.Flatten(start_dim=1, end_dim=-1),
        nn.Dropout(p=0.5, inplace=True),
        nn.Linear(num_features, 512),
        nn.GELU(),
        nn.Dropout(p=0.5, inplace=True),
        nn.Linear(512, 2)
    )
    return model

class DinoResNetFusion(nn.Module):
    def __init__(self, n_outputs=2):
        super(DinoResNetFusion, self).__init__()
        self.dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
        self.resnet = models.resnet50(weights='DEFAULT')
        
        # Freeze both backbones
        for param in self.dino.parameters():
            param.requires_grad = False
        for param in self.resnet.parameters():
            param.requires_grad = False
            
        # Get feature sizes
        dino_features = self.dino.embed_dim
        resnet_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity() # Remove ResNet's final layer
        
        # Fusion head
        self.fusion_head = RegressionHead(in_features=dino_features + resnet_features, n_outputs=n_outputs)

    def forward(self, x):
        # DINO features
        dino_out = self.dino.forward_features(x)
        dino_features = dino_out['x_norm_clstoken']
        
        # ResNet features
        resnet_features = self.resnet(x)
        
        # Concatenate features
        combined_features = torch.cat((dino_features, resnet_features), dim=1)
        
        # Pass through fusion head
        return self.fusion_head(combined_features)
    
class GpsSwinV2Model(nn.Module):
    """
    A wrapper class that combines the Swin Transformer V2 base model
    with a custom regression head.
    """
    def __init__(self, base_model, num_features):
        super(GpsSwinV2Model, self).__init__()
        self.swin_v2 = base_model # Use the V2 model
        self.head = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(num_features, 512),
            nn.GELU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, 2)
        )
            
    def forward(self, x):
        outputs = self.swin_v2(x)
        pooled_output = outputs.pooler_output
        return self.head(pooled_output)

def build_swin_v2_model():
    # --- KEY CHANGE 2 ---
    # Load the pre-trained Swin V2 base model
    model_name = "timm/swin_small_patch4_window7_224.ms_in22k_ft_in1k"
    base_model = Swinv2Model.from_pretrained(model_name)
    # --- END KEY CHANGE ---
    
    # Freeze the base model's parameters
    for param in base_model.parameters():
        param.requires_grad = False
        
    num_features = base_model.config.hidden_size # (This is 1024 for SwinV2-Base)
    
    model = GpsSwinV2Model(base_model, num_features)
    return model

def build_dinov3_model():
    # Load the pre-trained DINOv2 base model from Hugging Face
    model_path = r"D:\UPenn\CIS-5190\project\models\dinov3"
    # tokenizer = AutoTokenizer.from_pretrained(model_path)
    base_model = AutoModel.from_pretrained(model_path)
    
    # Freeze the base model's parameters (we will only train the head)
    for param in base_model.parameters():
        param.requires_grad = False
        
    # Get the number of features from the DINOv2 config
    num_features = base_model.config.hidden_size # (This is 768 for DINOv2-Base)
    
    # Create our full model
    model = GpsDinoModel(base_model, num_features)
    return model

def get_model(model_name):
    """Helper function to build the correct model architecture."""
    if model_name == 'efficientnet':
        print("Building EfficientNet-v2s structure...")
        return build_efficientnet_v2s()
    elif model_name == 'resnet':
        print("Building ResNet-50 structure...")
        return build_resnet50()
    elif model_name == 'dinov2':
        print("Building DINOv2 structure...")
        return build_dinov2_model()
    elif model_name == 'fusion':
        print("Building DINO-ResNet Fusion structure...")
        return DinoResNetFusion()
    elif model_name == 'convnext':
        print("Building OpenAI ViT structure...")
        return build_convexnet_model()
    elif model_name == 'swinv2':
        print("Building Swin Transformer V2 structure...")
        return build_swin_v2_model()
    elif model_name == 'dinov3':
        print("Building DINOv3 structure...")
        return build_dinov3_model()
    elif model_name == 'convnextv2':
        print("Building ConvNext V2 structure...")
        return build_convnextv2_model()
    else:
        raise ValueError(f"Unknown model_name: {model_name}")

# --- 3. Define Preprocessing ---
# Must be IDENTICAL to the validation transform in your training script.
val_transform = A.Compose([
    A.Resize(IMG_HEIGHT, IMG_WIDTH),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

# --- 4. Prediction Function ---
def predict_gps(image_path, model, scaler, transform, device):
    """Takes an image file path and returns the predicted [latitude, longitude]."""
    try:
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        processed_image = transform(image=image)['image']
        image_batch = processed_image.unsqueeze(0).to(device)
        
        with torch.no_grad():
            scaled_pred = model(image_batch)
        
        scaled_pred_np = scaled_pred.cpu().numpy()
        
        # FIX for the scikit-learn UserWarning
        # Use the column names from your metadata.csv
        scaled_pred_df = pd.DataFrame(scaled_pred_np, columns=['Latitude', 'Longitude'])
        real_gps_pred = scaler.inverse_transform(scaled_pred_df)
        
        return real_gps_pred[0]
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

# --- 5. Validation Loop ---
def evaluate_model_performance(model, scaler, transform, device):
    """Calculates the average real-world error in meters."""
    val_metadata_path = os.path.join(VAL_DIR, "metadata.csv")
    try:
        df = pd.read_csv(val_metadata_path)
        # Handle the column name from your CSV
        if 'file_name' in df.columns:
            df = df.rename(columns={'file_name': 'filename'})
            
    except FileNotFoundError:
        print(f"Error: metadata.csv not found in {VAL_DIR}")
        return
    
    errors = []
    
    print("\nStarting validation...")
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Validating"):
        # Use the column name from your CSV ('file_name' or 'filename')
        img_path = os.path.join(VAL_DIR, row['filename'])
        
        # Use the column names from your CSV ('Latitude', 'Longitude')
        true_coords = (row['Latitude'], row['Longitude'])
        
        pred_coords_arr = predict_gps(img_path, model, scaler, transform, DEVICE)
        
        if pred_coords_arr is not None:
            pred_coords = (pred_coords_arr[0], pred_coords_arr[1])
            distance = hs.haversine(true_coords, pred_coords, unit=hs.Unit.METERS)
            errors.append(distance)
            
    avg_error = np.mean(errors)
    print(f"\n--- ✅ Model Validation Complete ---")
    print(f"Average Real-World Prediction Error: {avg_error:.2f} meters")
    return avg_error

class GpsConvNextModelV2(nn.Module):
    """
    A wrapper class that combines the ConvNextV2 base model
    with a custom regression head.
    """
    def __init__(self, base_model, num_features):
        super(GpsConvNextModelV2, self).__init__()
        self.convnext = base_model # Use the ConvNextV2 model
        self.head = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(num_features, 512),
            nn.GELU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, 2)
        )
            
    def forward(self, x):
        outputs = self.convnext(x)
        # Just like Swin, ConvNext uses a 'pooler_output'
        pooled_output = outputs.pooler_output
        return self.head(pooled_output)

def build_convnextv2_model():
    # --- KEY CHANGE 2 ---
    # Load the pre-trained ConvNext V2 Tiny model
    model_name = "facebook/convnextv2-tiny-1k-224"
    base_model = ConvNextV2Model.from_pretrained(model_name)
    # --- END KEY CHANGE ---
    
    # Freeze the base model's parameters
    for param in base_model.parameters(): 
        param.requires_grad = False
        
    num_features = base_model.config.hidden_sizes[-1] # (This is 768, same as DINOv2-Base)
    
    model = GpsConvNextModelV2(base_model, num_features)
    return model

# --- 6. Main Script Execution ---
if __name__ == "__main__":
    # Setup the argument parser
    parser = argparse.ArgumentParser(description="Validate a GPS prediction model.")
    parser.add_argument(
        "--model_name", 
        type=str, 
        required=True, 
        choices=['efficientnet', 'resnet', 'dinov2', 'fusion','convnext','swinv2','dinov3','convnextv2'],
        help="The architecture of the model to load."
    )
    parser.add_argument(
        "--model_path", 
        type=str, 
        required=True, 
        help="Path to the saved .pth model file."
    )
    args = parser.parse_args()

    # --- Load Scaler ---
    if not os.path.exists(SCALER_PATH):
        print(f"Error: Scaler file not found at {SCALER_PATH}")
        print("Please run your training script first to create the scaler.")
        exit()
    scaler = joblib.load(SCALER_PATH)
    print(f"Scaler loaded from {SCALER_PATH}")

    # --- Load Model ---
    model = get_model(args.model_name)
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found at {args.model_path}")
        exit()
    
    # Load the saved weights into the model structure
    model.load_state_dict(torch.load(args.model_path, map_location=torch.device(DEVICE)))
    model.to(DEVICE)
    model.eval() # IMPORTANT: Set to evaluation mode
    print(f"Model loaded successfully from {args.model_path}")
    
    # --- Run Evaluation ---
    evaluate_model_performance(model, scaler, val_transform, DEVICE)