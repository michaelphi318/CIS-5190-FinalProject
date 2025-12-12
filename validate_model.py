# File: evaluate_with_ta_script.py
# This script loads your trained models and runs the TA's evaluation loop.

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from transformers import (
    AutoModel, Dinov2Model, Swinv2Model, ConvNextV2Model, SwinForImageClassification, BeitModel, ViTModel
)
import pandas as pd
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import os
import argparse
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import timm
from geopy.distance import geodesic

# --- 1. Configuration ---
VAL_DIR = r"D:\UPenn\CIS-5190\project\data\validation"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32 # You can make this larger for validation

# --- 2. Model Definitions ---
# (Contains all 8 of your model architectures)

def build_efficientnet_v2s():
    model = models.efficientnet_v2_s(weights='DEFAULT')
    for param in model.features.parameters():
        param.requires_grad = False
    
    num_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Linear(num_features, 512),
        nn.GELU(),
        nn.Dropout(p=0.3, inplace=False),
        nn.Linear(512, 2)
    )
    return model
def build_resnet50():
    model = models.resnet50(weights=None)
    num_features = model.fc.in_features
    model.fc = nn.Sequential(nn.Linear(num_features, 512), nn.ReLU(), nn.Dropout(0.5), nn.Linear(512, 2))
    return model

def build_resnet18_model():
    model = models.resnet18(weights=None)
    num_features = model.fc.in_features
    model.fc = nn.Sequential(nn.Dropout(p=0.5), nn.Linear(num_features, 256), nn.ReLU(), nn.Dropout(p=0.5), nn.Linear(256, 2))
    return model
    
class GpsDinoModel(nn.Module):
    def __init__(self, base_model, num_features):
        super(GpsDinoModel, self).__init__()
        self.dino = base_model
        self.head = nn.Sequential( nn.Linear(num_features, 512),
            nn.GELU(), nn.Dropout(p=0.3), nn.Linear(512, 2))
    def forward(self, x):
        outputs = self.dino(x)
        cls_token = outputs.last_hidden_state[:, 0]
        return self.head(cls_token)

def build_dinov2_model():
    base_model = Dinov2Model.from_pretrained("facebook/dinov2-base")
    num_features = base_model.config.hidden_size
    model = GpsDinoModel(base_model, num_features)
    return model

class GpsSwinV2Model(nn.Module):
    def __init__(self, base_model, num_features):
        super(GpsSwinV2Model, self).__init__()
        self.swin_v2 = base_model
        self.head = nn.Sequential(nn.Dropout(p=0.5), nn.Linear(num_features, 512), nn.GELU(), nn.Dropout(p=0.5), nn.Linear(512, 2))
    def forward(self, x):
        outputs = self.swin_v2(x); return self.head(outputs.pooler_output)

class GpsBeitModel(nn.Module):
    """
    A wrapper class that combines the BEiT base model with a custom
    regression head.
    """
    def __init__(self, base_model, num_features):
        super(GpsBeitModel, self).__init__()
        self.beit = base_model
        # Using the same head as your successful DINO model
        self.head = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.GELU(),
            nn.Dropout(p=0.3),
            nn.Linear(512, 2)
        )
            
    def forward(self, x):
        # Pass the input through the BEiT base
        outputs = self.beit(x)
        
        # BEiT (like Swin) uses 'pooler_output'
        pooled_output = outputs.pooler_output
        
        # Pass this single feature vector through our regression head
        return self.head(pooled_output)

def build_beit_model():
    # --- KEY CHANGE 2 ---
    # Load the pre-trained BEiT base model from Hugging Face
    model_name = "microsoft/beit-base-patch16-224"
    base_model = BeitModel.from_pretrained(model_name)
    # --- END KEY CHANGE ---
    
    # Freeze the base model's parameters
    for param in base_model.parameters():
        param.requires_grad = False
        
    # Get the number of features from the BEiT config
    num_features = base_model.config.hidden_size # (This is 768, same as DINOv2-Base)
    
    # Create our full model
    model = GpsBeitModel(base_model, num_features)
    return model

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
    #tokenizer = AutoTokenizer.from_pretrained(model_path)
    base_model = AutoModel.from_pretrained(model_path)
    
    # Freeze the base model's parameters (we will only train the head)
    for param in base_model.parameters():
        param.requires_grad = False
        
    # Get the number of features from the DINOv2 config
    num_features = base_model.config.hidden_size # (This is 768 for DINOv2-Base)
    
    # Create our full model
    model = GpsDinoModel(base_model, num_features)
    return model

# --- 4. Model Definition (using timm - ROBUST VERSION) ---
class GpsTimmModel(nn.Module):
    """
    A wrapper class that combines a timm base model
    with a custom regression head.
    
    This version loads the backbone as a feature extractor (no pooling)
    and applies its own pooling, which is more robust.
    """
    def __init__(self, model_name, gps_mean, gps_std, n_outputs=2):
        super(GpsTimmModel, self).__init__()
        
        # 1. Load the timm model as a feature extractor
        #    global_pool='' gets the raw [B, C, H, W] feature maps
        self.backbone = timm.create_model(
            model_name,
            pretrained=True,
            num_classes=0,       # Remove the original classifier
            global_pool=''       # <-- We will do our own pooling
        )
        
        # 2. Get the feature size from the timm model
        num_features = self.backbone.num_features

        # 3. Create our own pooling layer
        #    This will take the [B, C, 7, 7] map and turn it into [B, C, 1, 1]
        self.pooling = nn.AdaptiveAvgPool2d(1)

        # 4. Create our new head (for 10k+ images)
        self.head = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.GELU(),
            nn.Dropout(p=0.2),
            nn.Linear(512, n_outputs)
        )
        self.register_buffer("gps_mean", gps_mean)
        self.register_buffer("gps_std", gps_std)

    def forward(self, x):
        # 1. Get feature maps: [Batch_Size, Num_Features, H, W]
        #    e.g., [7, 1024, 7, 7]
        feature_maps = self.backbone(x)
        
        # 2. Apply our own pooling: [7, 1024, 1, 1]
        pooled_features = self.pooling(feature_maps)
        
        # 3. Flatten the features: [7, 1024]
        #    This is the step you are likely missing!
        flattened_features = torch.flatten(pooled_features, 1)
        
        # 4. Pass through the head: [7, 2]
        return self.head(flattened_features)

def build_dinov3_convnext_model(gps_mean, gps_std):

    model_name = "convnext_base.dinov3_lvd1689m"
    model = GpsTimmModel(model_name,gps_mean, gps_std)

    # Freeze the backbone parameters
    for param in model.backbone.parameters():
        param.requires_grad = False
        
    return model
# --- End of Model Definition ---
class GpsConvNextModelV2(nn.Module):
    """
    A wrapper class that combines the ConvNextV2 base model
    with a custom regression head.
    """
    def __init__(self, base_model, num_features):
        super(GpsConvNextModelV2, self).__init__()
        self.convnext = base_model # Use the ConvNextV2 model
        self.head = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.GELU(),
            nn.Dropout(p=0.3),
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
    model_name = "facebook/convnextv2-base-22k-224"
    base_model = ConvNextV2Model.from_pretrained(model_name)
    # --- END KEY CHANGE ---
    
    # Freeze the base model's parameters
    for param in base_model.parameters(): # <-- THIS IS THE FIX
        param.requires_grad = False
        
    num_features = base_model.config.hidden_sizes[-1] # (This is 768, same as DINOv2-Base)
    
    model = GpsConvNextModelV2(base_model, num_features)
    return model

# --- 5. Training & Validation Functions---
def train_one_epoch(model, dataloader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0.0
    for images, labels in tqdm(dataloader, desc="Training"):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
    return total_loss / len(dataloader.dataset)

class GpsViTModel(nn.Module):
    """
    A wrapper class that combines the Google ViT base model with a custom
    regression head.
    """
    def __init__(self, base_model, num_features):
        super(GpsViTModel, self).__init__()
        self.vit = base_model
        # Using the same head as your successful DINO model
        self.head = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.GELU(),
            nn.Dropout(p=0.3), # Using your 10k+ data dropout
            nn.Linear(512, 2)
        )
            
    def forward(self, x):
        # Pass the input through the ViT base
        outputs = self.vit(x)
        
        # Standard ViT (like BEiT/Swin) uses 'pooler_output'
        # This is the [CLS] token after passing through a Linear layer and Tanh
        pooled_output = outputs.pooler_output
        
        # Pass this single feature vector through our regression head
        return self.head(pooled_output)

def build_google_vit_model():
    # --- KEY CHANGE 2 ---
    # Load the pre-trained Google ViT base model from Hugging Face
    model_name = "google/vit-base-patch16-224-in21k"
    base_model = ViTModel.from_pretrained(model_name)
    # --- END KEY CHANGE ---
    
    # Freeze the base model's parameters
    for param in base_model.parameters():
        param.requires_grad = False
        
    # Get the number of features from the ViT config
    num_features = base_model.config.hidden_size # (This is 768, same as DINOv2-Base)
    
    # Create our full model
    model = GpsViTModel(base_model, num_features)
    return model

class GpsManualFusion(nn.Module):
    def __init__(self, n_outputs=2):
        super(GpsManualFusion, self).__init__()
        
        # --- Branch 1: DINOv2 (ViT) ---
        self.dino_backbone = Dinov2Model.from_pretrained("facebook/dinov2-base")
        dino_features = self.dino_backbone.config.hidden_size # 768
        
        # --- Branch 2: ConvNextV2 (CNN) ---
        self.cnn_backbone = ConvNextV2Model.from_pretrained("facebook/convnextv2-base-22k-224")
        cnn_features = self.cnn_backbone.config.hidden_sizes[-1] # 1024
        
        # Freeze both backbones
        for param in self.dino_backbone.parameters():
            param.requires_grad = False
        for param in self.cnn_backbone.parameters():
            param.requires_grad = False
            
        # --- Fusion Head ---
        total_features = dino_features + cnn_features # 768 + 1024 = 1792
        # Using the head from your DINO script (p=0.3 dropout)
        self.head = nn.Sequential(
            nn.Linear(total_features, 512),
            nn.GELU(),
            nn.Dropout(p=0.3),
            nn.Linear(512, n_outputs)
        )

    def forward(self, x):
        # --- Process DINO Branch ---
        dino_out = self.dino_backbone(x)
        dino_vec = dino_out.last_hidden_state[:, 0] # [B, 768]
        
        # --- Process CNN Branch ---
        cnn_out = self.cnn_backbone(x)
        cnn_vec = cnn_out.pooler_output # [B, 1024]
        
        # --- Concatenate Features ---
        combined_vec = torch.cat((dino_vec, cnn_vec), dim=1) # [B, 1792]
        
        # --- Pass through Head ---
        return self.head(combined_vec)

def build_fusion_model():
    model = GpsManualFusion()
    # Note: Backbones are frozen *inside* the class
    return model

def get_model(model_name):
    STAT_Path = r"D:\UPenn\CIS-5190\project\models\gps_stats.pt"
    stats = torch.load(STAT_Path)
    gps_mean = stats['mean']
    gps_std = stats['std']
    if model_name == 'efficientnet': return build_efficientnet_v2s()
    elif model_name == 'resnet18': return build_resnet18_model()
    elif model_name == 'resnet50': return build_resnet50()
    elif model_name == 'dinov2': return build_dinov2_model()
    elif model_name == 'swinv2': return build_swin_v2_model()
    elif model_name == 'dinov3': return build_dinov3_model()
    elif model_name == 'convnextv2': return build_convnextv2_model()
    elif model_name == 'dinov3_convnext': return build_dinov3_convnext_model(gps_mean, gps_std)
    elif model_name == 'beit': return build_beit_model()
    elif model_name == 'google_vit': return build_google_vit_model()
    elif model_name == 'fusion': return build_fusion_model()
    # Add other models like 'fusion' or 'convnext' if you need them
    else: raise ValueError(f"Unknown model_name: {model_name}")


# --- 3. Data Loading ---

# Define the dataset class from your training script
class CampusDataset(Dataset):
    def __init__(self, df, image_dir, gps_mean, gps_std, transform=None):
        self.df = df
        self.image_dir = image_dir
        self.gps_mean = gps_mean.to(torch.float32)
        self.gps_std = gps_std.to(torch.float32)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = os.path.join(self.image_dir, row['file_name'])
        
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            image = self.transform(image=image)['image']
        
        gps_coords = row[['Latitude', 'Longitude']].values.astype(np.float32)
        label_tensor = torch.from_numpy(gps_coords)
        
        # Normalize label
        scaled_label = (label_tensor - self.gps_mean) / self.gps_std
        
        # --- CHANGE: Return filename as well ---
        return image, scaled_label, row['file_name']

# Define the validation transform
val_transform = A.Compose([
    A.Resize(IMG_HEIGHT, IMG_WIDTH),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])


# --- 4. Main Script Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TA's Evaluation Script for Custom Models.")
    parser.add_argument("--model_name", type=str, required=True, 
                        choices=['efficientnet', 'resnet18', 'resnet50', 'dinov2', 'swinv2', 'dinov3', 'convnextv2','dinov3_convnext','beit','google_vit','fusion'],
                        help="The architecture of the model to load.")
    parser.add_argument("--model_path", type=str, required=True, 
                        help="Path to the saved .pth model file.")
    parser.add_argument("--stats_path", type=str, required=False, 
                        help="Path to the corresponding .pt stats (mean/std) file.")
    args = parser.parse_args()
    STAT_Path = r"D:\UPenn\CIS-5190\project\models\gps_stats.pt"
    # --- Load Stats (mean/std) ---
    if not args.stats_path:
        print(f"Using default stats path: {STAT_Path}")
        STAT_Path = STAT_Path
    else:
        STAT_Path = args.stats_path
    stats = torch.load(STAT_Path)
    gps_mean = stats['mean']
    gps_std = stats['std']
    all_filenames = []
    # --- Load Model ---
    model = get_model(args.model_name)
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found at {args.model_path}")
        exit()
    
    model.load_state_dict(torch.load(args.model_path, map_location=torch.device(DEVICE)))
    model.to(DEVICE)
    model.eval() # IMPORTANT: Set to evaluation mode
    print(f"Model loaded successfully from {args.model_path}")

    # --- Create Validation DataLoader ---
    val_df = pd.read_csv(os.path.join(VAL_DIR, "metadata.csv"))
    val_dataset = CampusDataset(df=val_df, image_dir=VAL_DIR, 
                                gps_mean=gps_mean, gps_std=gps_std, 
                                transform=val_transform)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    print("Validation DataLoader created.")

    # --- TA's EVALUATION SCRIPT (MODIFIED) ---
    
    # Move mean/std to CPU for de-normalization
    lat_mean = gps_mean[0].item()
    lat_std = gps_std[0].item()
    lon_mean = gps_mean[1].item()
    lon_std = gps_std[1].item()
    
    all_preds_scaled = [] # Scaled predictions (Z-scores)
    all_actuals_scaled = [] # Scaled actuals (Z-scores)
    all_preds_denorm = [] # De-normalized (real GPS)
    all_actuals_denorm = [] # De-normalized (real GPS)
    distances = []

    model.eval() # Use the generic 'model' variable
    with torch.no_grad():
        for images, gps_coords_scaled,filenames in tqdm(val_dataloader, desc="Running TA Evaluation"):
            images = images.to(DEVICE)
            gps_coords_scaled = gps_coords_scaled.to(DEVICE)

            outputs_scaled = model(images) # Model predicts scaled Z-scores
            all_filenames.extend(filenames)
            # De-normalize predictions and actual values
            # (pred * std) + mean
            preds_denorm = outputs_scaled.cpu() * torch.tensor([lat_std, lon_std]) + torch.tensor([lat_mean, lon_mean])
            actuals_denorm = gps_coords_scaled.cpu() * torch.tensor([lat_std, lon_std]) + torch.tensor([lat_mean, lon_mean])

            # Store both scaled and de-normalized for metrics
            all_preds_scaled.append(outputs_scaled.cpu())
            all_actuals_scaled.append(gps_coords_scaled.cpu())
            all_preds_denorm.append(preds_denorm)
            all_actuals_denorm.append(actuals_denorm)

            # Calculate Haversine distance
            for pred, actual in zip(preds_denorm, actuals_denorm):
                distance = geodesic((actual[0], actual[1]), (pred[0], pred[1])).meters
                distances.append(distance)


# Concatenate all batches
    all_preds_scaled = torch.cat(all_preds_scaled).numpy()
    all_actuals_scaled = torch.cat(all_actuals_scaled).numpy()
    all_preds_denorm = torch.cat(all_preds_denorm).numpy()
    all_actuals_denorm = torch.cat(all_actuals_denorm).numpy()

    # --- Compute Error Metrics (as in TA's script) ---
    
    # 1. Scaled error (what your val_loss was)
    mae_scaled = mean_absolute_error(all_actuals_scaled, all_preds_scaled)
    mse_scaled = mean_squared_error(all_actuals_scaled, all_preds_scaled) # Get MSE
    rmse_scaled = np.sqrt(mse_scaled)
    print("\n--- SCALED Metrics (Z-score) ---")
    print(f'Scaled Mean Absolute Error: {mae_scaled:.6f}')
    print(f'Scaled Root Mean Squared Error: {rmse_scaled:.6f}')

    # 2. De-normalized error (real GPS error, in degrees)
    mae_degrees = mean_absolute_error(all_actuals_denorm, all_preds_denorm)
    mse_degrees = mean_squared_error(all_actuals_denorm, all_preds_denorm) # Get MSE
    rmse_degrees = np.sqrt(mse_degrees)
    print("\n--- DE-NORMALIZED Metrics (Degrees) ---")
    print(f'De-normalized Mean Absolute Error: {mae_degrees:.6f} (degrees)')
    print(f'De-normalized Root Mean Squared Error: {rmse_degrees:.6f} (degrees)')
    
    # 3. Real-world distance error (The one that matters)
    avg_distance = sum(distances) / len(distances)
    print("\n--- REAL-WORLD Metrics (Meters) ---")
    print(f"Avg Distance: {avg_distance:.2f} meters")

    # --- TA's PLOTTING SCRIPT ---
    print("\nGenerating plot...")
    plt.figure(figsize=(10, 5))

    # Plot actual points
    plt.scatter(all_actuals_denorm[:, 1], all_actuals_denorm[:, 0], label='Actual', color='blue', alpha=0.6)

    # Plot predicted points
    plt.scatter(all_preds_denorm[:, 1], all_preds_denorm[:, 0], label='Predicted', color='red', alpha=0.6)

    # Draw lines connecting actual and predicted points
    for i in range(len(all_actuals_denorm)):
        plt.plot(
            [all_actuals_denorm[i, 1], all_preds_denorm[i, 1]],
            [all_actuals_denorm[i, 0], all_preds_denorm[i, 0]],
            color='gray', linewidth=0.5
        )

    plt.legend()
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Actual vs. Predicted GPS Coordinates with Error Lines')
    plt.show()

    # --- NEW SECTION: Top 50 Worst Errors ---
    print("\n" + "="*60)
    print(" TOP 50 WORST PREDICTIONS (Check these for bad data!)")
    print("="*60)

    # Create a DataFrame for easy sorting and viewing
    error_df = pd.DataFrame({
        'filename': all_filenames,
        'error_meters': distances,
        'pred_lat': all_preds_denorm[:, 0],
        'pred_lon': all_preds_denorm[:, 1],
        'actual_lat': all_actuals_denorm[:, 0],
        'actual_lon': all_actuals_denorm[:, 1]
    })

    # Sort by error (descending)
    worst_offenders = error_df.sort_values(by='error_meters', ascending=False).head(50)

    # Print cleanly
    pd.set_option('display.max_rows', 50)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print(worst_offenders[['filename', 'error_meters', 'actual_lat', 'actual_lon']])

    # Save to CSV for deeper analysis in Excel
    worst_offenders.to_csv("worst_errors.csv", index=False)
    print("\nSaved top 50 worst errors to 'worst_errors.csv'")

    print("Done.")
