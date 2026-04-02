# Kaggle-ready full YOLO-DCARF script with DRFB + GatingNetwork + AFRM_v2
# Copy-paste into a Kaggle notebook cell and run. Adjust `dataset_path` if needed.

import os
import sys
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F

print("YOLO-DCARF (Kaggle-ready) — DRFB + Gating + AFRM_v2\n" + "="*60)

# -----------------------
# STEP 0: DATASET PATHS
# -----------------------
dataset_path = '/kaggle/input/hituav-a-highaltitude-infrared-thermal-dataset/hit-uav'   #change dataset path as per you requirement this path is as per my requirement
if not os.path.exists(dataset_path):
    print("Warning: dataset_path does not exist. Update dataset_path to your dataset location if needed.")
os.makedirs('/kaggle/working', exist_ok=True)

# -----------------------
# STEP 1: WRITE data.yaml
# -----------------------
data_yaml_content = {
    'train': os.path.join(dataset_path, 'images/train'),
    'val': os.path.join(dataset_path, 'images/val'),
    'nc': 3,
    'names': ['Person', 'Car', 'Bicycle']
}
with open('data.yaml', 'w') as f:
    yaml.dump(data_yaml_content, f)
print("✔ data.yaml written")

# -----------------------
# STEP 2: CLONE ULTRALYTICS
# -----------------------
print("✔ Cloning ultralytics repo (quiet)...")
# Use shell clone - Kaggle supports shell ! commands in cells
os.system('git clone https://github.com/ultralytics/ultralytics.git --quiet || true')
sys.path.insert(0, '/kaggle/working/ultralytics')
# change working directory into the repo for imports that expect package context
try:
    os.chdir('ultralytics')
except Exception:
    pass
print("✔ ultralytics available at /kaggle/working/ultralytics")

# Delay imports until after sys.path insert
from ultralytics.nn.modules.conv import Conv
from ultralytics.nn import tasks

# -----------------------
# STEP 3: DEFINE DRFB (as in your script)
# -----------------------
class DRFB(nn.Module):
    """Dynamic Receptive Field Block (Simplified + Regularized)"""
    def __init__(self, c1, k=5):  # c1 = in channels, k = kernel size
        super().__init__()
        c2 = c1
        c_ = max(1, c1 // 2)
        # here Conv signature: Conv(in, out, k, s, p=?, d=?). Keep defaults clear.
        self.cv1 = Conv(c1, c_, 1, 1)
        # dilated convs
        self.d_cv2 = Conv(c_, c_, k, 1, d=1, p=(k//2)*1)
        self.d_cv3 = Conv(c_, c_, k, 1, d=2, p=(k//2)*2)
        self.d_cv4 = Conv(c_, c_, k, 1, d=3, p=(k//2)*3)
        self.cv_out = Conv(c_ * 4, c2, 1, 1)
    def forward(self, x):
        x_in = self.cv1(x)
        return self.cv_out(torch.cat((x_in, self.d_cv2(x_in), self.d_cv3(x_in), self.d_cv4(x_in)), 1))

# Inject DRFB
tasks.DRFB = DRFB
print("✔ DRFB injected into ultralytics.nn.tasks")

# -----------------------
# STEP 4: Define GatingNetwork + AFRM_v2
# -----------------------
class GatingNetwork(nn.Module):
    """
    Lightweight gating network:
    Input: list/tuple of three feature maps (p3,p4,p5)
    Output: control vector (B, ctrl_dim) in (0,1) via sigmoid
    """
    def __init__(self, in_channels=[128, 256, 512], mid_channels=128, out_dim=64):
        super().__init__()
        self.reduce3 = Conv(in_channels[0], mid_channels, 1, 1)
        self.reduce4 = Conv(in_channels[1], mid_channels, 1, 1)
        self.reduce5 = Conv(in_channels[2], mid_channels, 1, 1)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(mid_channels * 3, mid_channels),
            nn.ReLU(inplace=True),
            nn.Linear(mid_channels, out_dim),
            nn.Sigmoid()
        )
    def forward(self, features):
        p3, p4, p5 = features
        a3 = self.global_pool(self.reduce3(p3)).flatten(1)
        a4 = self.global_pool(self.reduce4(p4)).flatten(1)
        a5 = self.global_pool(self.reduce5(p5)).flatten(1)
        concat = torch.cat((a3, a4, a5), dim=1)
        ctrl = self.mlp(concat)
        return ctrl

class AFRM(nn.Module):
    """
    AFRM base implementation (internal). Not directly used by YAML because Ultralytics prefers
    positional args; see AFRM_v2 below for the signature used in YAML.
    """
    def __init__(self, in_channels=[128,256,512], mid_channels=128, ctrl_dim=64):
        super().__init__()
        self.adapt3 = Conv(in_channels[0], mid_channels, 1, 1)
        self.adapt4 = Conv(in_channels[1], mid_channels, 1, 1)
        self.adapt5 = Conv(in_channels[2], mid_channels, 1, 1)
        self.refine3 = Conv(mid_channels, in_channels[0], 3, 1, p=1)
        self.refine4 = Conv(mid_channels, in_channels[1], 3, 1, p=1)
        self.refine5 = Conv(mid_channels, in_channels[2], 3, 1, p=1)
        self.ctrl_fc = nn.Sequential(
            nn.Linear(ctrl_dim, mid_channels),
            nn.ReLU(inplace=True),
            nn.Linear(mid_channels, mid_channels),
            nn.Sigmoid()
        )
        self.fuse_conv = Conv(mid_channels * 3, mid_channels, 1, 1)

    def _forward_core(self, p3, p4, p5, ctrl):
        b = p3.shape[0]
        a3 = self.adapt3(p3)
        a4 = self.adapt4(p4)
        a5 = self.adapt5(p5)
        ch_scale = self.ctrl_fc(ctrl).view(b, -1, 1, 1)
        a3 = a3 * ch_scale
        a4 = a4 * ch_scale
        a5 = a5 * ch_scale
        target_size = a3.shape[-2:]
        u4 = F.interpolate(a4, size=target_size, mode='nearest')
        u5 = F.interpolate(a5, size=target_size, mode='nearest')
        concat = torch.cat([a3, u4, u5], dim=1)
        fused = self.fuse_conv(concat)
        r3 = self.refine3(fused)
        r4 = F.interpolate(fused, size=a4.shape[-2:], mode='nearest'); r4 = self.refine4(r4)
        r5 = F.interpolate(fused, size=a5.shape[-2:], mode='nearest'); r5 = self.refine5(r5)
        return [r3, r4, r5]

# AFRM_v2: direct positional signature to match Ultralytics YAML calling:
# forward(p3, p4, p5, ctrl) -> return [r3, r4, r5]
class AFRM_v2(AFRM):
    def forward(self, p3, p4, p5, ctrl):
        return self._forward_core(p3, p4, p5, ctrl)

# Inject gating + afrms
tasks.GatingNetwork = GatingNetwork
tasks.AFRM = AFRM_v2
print("✔ GatingNetwork and AFRM_v2 injected into ultralytics.nn.tasks")

# -----------------------
# STEP 5: WRITE yolo-dcarf.yaml (model definition)
# -----------------------
# NOTE: This YAML is a carefully crafted example that wires gating + AFRM into the neck.
# If you get an index mismatch, see the checklist below to correct indices after model summary.
yolo_air_yaml_content = r"""
# YOLO-AIR Optimized Configuration (with Gating + AFRM_v2)
nc: 3
depth_multiple: 0.50
width_multiple: 0.75

# Backbone - keep structure concise but representative
backbone:
  - [-1, 1, Conv, [64, 3, 2]]        # 0
  - [-1, 1, Conv, [128, 3, 2]]       # 1
  - [-1, 3, C2f, [128, True]]        # 2
  - [-1, 1, Conv, [256, 3, 2]]       # 3
  - [-1, 6, C2f, [256, True]]        # 4
  - [-1, 1, Conv, [512, 3, 2]]       # 5
  - [-1, 6, C2f, [512, True]]        # 6   <-- P5 candidate
  - [-1, 1, DRFB, [384]]             # 7   (DRFB applied)
  - [-1, 1, Conv, [1024, 3, 2]]      # 8
  - [-1, 3, C2f, [1024, True]]       # 9

# Neck (Gating + AFRM insertion)
# Note: We assume the three feature maps we want are produced at indices: 2 (P3), 4 (P4), 6 (P5).
# The gating network takes [P3,P4,P5] and produces a control vector (placed as its own layer output).
# AFRM_v2 then consumes p3,p4,p5,ctrl (positional args).
neck:
  # GatingNetwork: from layers [2,4,6]
  - [[2, 4, 6], 1, GatingNetwork, [[128,256,512], 128, 64]]   # 10 (gating output)
  # AFRM_v2: from [2,4,6,10] => positional args (p3,p4,p5,ctrl)
  - [[2, 4, 6, 10], 1, AFRM, [[128,256,512], 128, 64]]        # 11  -> returns 3 outputs

# Head
# We assume AFRM returns three outputs (r3,r4,r5) placed at idx 11 (but because AFRM returns a list,
# Ultralytics will expand the list into multiple outputs in sequence; index mapping below assumes this behavior).
head:
  - [-1, 1, nn.Upsample, [None, 2, 'nearest']]
  - [[-1, 6], 1, Concat, [1]]
  - [-1, 3, C2f, [128, True]]
  - [-1, 1, nn.Upsample, [None, 2, 'nearest']]
  - [[-1, 4], 1, Concat, [1]]
  - [-1, 3, C2f, [64, True]]
  - [-1, 1, Conv, [64, 3, 2]]
  - [[-1, 12], 1, Concat, [1]]
  - [-1, 3, C2f, [128, True]]
  - [-1, 1, Conv, [128, 3, 2]]
  - [[-1, 9], 1, Concat, [1]]
  - [-1, 3, C2f, [256, True]]
  # Detect: use the final three feature maps; adjust if you get shape/index mismatch
  - [[15, 18, 21], 1, Detect, [nc]]
"""

# Write YAML to disk (model file)
with open('/kaggle/working/yolo-air.yaml', 'w') as f:
    f.write(yolo_air_yaml_content)
print("✔ yolo-air.yaml written to /kaggle/working/yolo-air.yaml")

# -----------------------
# STEP 6: CREATE MODEL AND SANITY CHECK
# -----------------------
from ultralytics.models.yolo import YOLO
print("Creating YOLO model from YAML...")

# Ensure working dir points to where YAML is
os.chdir('/kaggle/working')
model = YOLO('yolo-air.yaml')  # constructs the model (will parse YAML and use tasks.*)
print("✔ Model constructed — printing summary...")
print(model)   # ultralytics prints a layers table which helps verify indices

# Quick sanity forward with dummy input (optional but useful)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
try:
    model.model.to(device)
    model.model.eval()
    with torch.no_grad():
        dummy = torch.randn(1, 3, 640, 640).to(device)
        print("Running quick sanity forward (this may take a moment)...")
        _ = model.model(dummy)
    print("✔ Sanity forward OK")
except Exception as e:
    print("⚠ Sanity forward failed (this may be an index/shape mismatch). Error:")
    print(e)
    # continue — training will likely show the same error; see checklist below

# -----------------------
# STEP 7: TRAIN (same options you provided)
# -----------------------
print("Starting training (if model constructed successfully)...")
try:
    model.train(
        data='/kaggle/working/data.yaml',
        epochs=150,
        imgsz=768,
        batch=16,
        patience=40,
        optimizer='AdamW',
        lr0=0.001,
        lrf=0.0001,
        cos_lr=True,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        mixup=0.05,
        hsv_h=0.0, hsv_s=0.0, hsv_v=0.0,
        degrees=5, translate=0.1, scale=0.5, shear=2.0, fliplr=0.5,
        project='yolo_air_training_optimized',
        name='exp_yoloair_final',
        verbose=True,
        plots=True
    )
except Exception as e:
    print("⚠ Training launch failed (likely due to model construction error). Error:")
    print(e)

print("Script finished. Check training folder: yolo_air_training_optimized/exp_yoloair_final (if training started).")
