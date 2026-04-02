import os
import yaml
import sys
import torch
import torch.nn as nn

print("YOLO-DCARF (Optimized Training Configuration)\n" + "=" * 45)

# -------------------------------------------------------------------------
# STEP 1: ENVIRONMENT AND DATASET CONFIGURATION
# -------------------------------------------------------------------------
print(" Step 1: Preparing environment...")

dataset_path = '/kaggle/input/hituav-a-highaltitude-infrared-thermal-dataset/hit-uav'

data_yaml_content = {
    'train': os.path.join(dataset_path, 'images/train'),
    'val': os.path.join(dataset_path, 'images/val'),
    'nc': 3,
    'names': ['Person', 'Car', 'Bicycle']
}

with open('data.yaml', 'w') as f:
    yaml.dump(data_yaml_content, f)
print("   - data.yaml created successfully.")
print("-" * 50)

# -------------------------------------------------------------------------
# STEP 2: CLONE ULTRALYTICS REPOSITORY
# -------------------------------------------------------------------------
print("Step 2: Cloning Ultralytics repository...")

!git clone https://github.com/ultralytics/ultralytics.git --quiet
sys.path.insert(0, '/kaggle/working/ultralytics')
os.chdir('ultralytics')
print("   - Cloned and moved into ultralytics directory.")
print("-" * 50)
import os
import yaml
from ultralytics import YOLO

print("YOLO Benchmark Training: v8m vs v5m\n" + "=" * 45)

# -------------------------------------------------------------------------
# STEP 1: ENSURE DATA.YAML EXISTS
# -------------------------------------------------------------------------
dataset_path = '/kaggle/input/hituav-a-highaltitude-infrared-thermal-dataset/hit-uav'
data_yaml_content = {
    'train': os.path.join(dataset_path, 'images/train'),
    'val': os.path.join(dataset_path, 'images/val'),
    'nc': 3,
    'names': ['Person', 'Car', 'Bicycle']
}

with open('data.yaml', 'w') as f:
    yaml.dump(data_yaml_content, f)
print("data.yaml ready.")

# -------------------------------------------------------------------------
# SHARED HYPERPARAMETERS (Identical to your YOLO-Air run)
# -------------------------------------------------------------------------
# We use the exact same settings to ensure the comparison is about the Architecture, not the config.
train_args = dict(
    data='data.yaml',
    epochs=150,
    imgsz=768,              # Large imgsz for UAV small objects
    batch=16,               # Adjust if you hit OOM (Try 12 or 8 for 'm' models)
    patience=40,
    optimizer='AdamW',
    lr0=0.001,
    lrf=0.0001,
    cos_lr=True,
    momentum=0.937,
    weight_decay=0.0005,
    warmup_epochs=3,
    mixup=0.05,
    degrees=5, translate=0.1, scale=0.5, shear=2.0, fliplr=0.5,
    hsv_h=0.0, hsv_s=0.0, hsv_v=0.0,
    verbose=True,
    plots=True
)

# -------------------------------------------------------------------------
# STEP 2: TRAIN YOLOv8m (Medium)
# -------------------------------------------------------------------------
print("\n" + "="*20 + " STARTING YOLOv8m " + "="*20)
# Load pretrained YOLOv8m
model_v8 = YOLO('yolov8m.pt') 

model_v8.train(
    project='benchmark_training',
    name='yolov8m_hituav',
    **train_args
)

# -------------------------------------------------------------------------
# STEP 3: TRAIN YOLOv8m (Medium)
# -------------------------------------------------------------------------
print("\n" + "="*20 + " STARTING YOLOv5m " + "="*20)


print("\n" + "="*50)
print("Benchmarking Complete!")
print("Check results in: /kaggle/working/benchmark_training/")
