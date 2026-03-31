import os
import cv2
import glob
import random
import pandas as pd
import matplotlib.pyplot as plt
from ultralytics import YOLO

print("✅ Results and Visualization Script Started.")
print("-" * 50)

# -------------------------------------------------------------------------
# STEP 1: LOCATE THE TRAINING RESULTS
# -------------------------------------------------------------------------
# The results are saved in the project directory defined during training
results_dir = '/kaggle/working/yolo_air_training_optimized/exp_yoloair_final'
weights_path = os.path.join(results_dir, 'weights/best.pt')

# Check if training is complete by looking for the weights file
if not os.path.exists(weights_path):
    print("⏳ Training is still in progress. Please wait for it to complete.")
    # You can add a loop here to wait, but it's better to run this cell after training finishes.
else:
    print(f"   - Found trained weights at: {weights_path}")

    # -------------------------------------------------------------------------
    # STEP 2: DISPLAY PERFORMANCE METRICS
    # -------------------------------------------------------------------------
    print("\n✅ Step 2: Displaying Final Performance Metrics...")
    results_csv_path = os.path.join(results_dir, 'results.csv')
    if os.path.exists(results_csv_path):
        results_df = pd.read_csv(results_csv_path)
        # Clean up column names by removing whitespace
        results_df.columns = results_df.columns.str.strip()
        print("   - Final metrics from the last epoch:")
        # Display the last row which contains the final results
        print(results_df.tail(1).to_string(index=False))
    else:
        print("   - Could not find results.csv file.")
    print("-" * 50)
# STEP 3: VISUALIZE GROUND TRUTH VS. PREDICTIONS (CORRECTED)
# -------------------------------------------------------------------------
print("\n Step 3: Visualizing Ground Truth vs. Model Predictions...")

# Load the best trained model
model = YOLO(weights_path)

# --- THE FIX IS HERE ---
# Find all common image types to be more robust
val_img_dir = '/kaggle/input/hituav-a-highaltitude-infrared-thermal-dataset/hit-uav/images/val'
val_label_dir = '/kaggle/input/hituav-a-highaltitude-infrared-thermal-dataset/hit-uav/labels/val'
image_paths = glob.glob(os.path.join(val_img_dir, '*.png'))
image_paths.extend(glob.glob(os.path.join(val_img_dir, '*.jpg')))
image_paths.extend(glob.glob(os.path.join(val_img_dir, '*.jpeg')))

# Check if any images were found
if not image_paths:
    print("   - ❌ ERROR: No images found in the validation directory. Please check the path.")
else:
    # Determine how many images to sample. It will be 7 or fewer if not enough images exist.
    num_to_sample = min(11, len(image_paths))
    print(f"   - Found {len(image_paths)} images. Selecting {num_to_sample} for visualization.")

    # Select the images for visualization
    selected_images = random.sample(image_paths, num_to_sample)

    # Define class names and colors for boxes
    class_names = ['Person', 'Car', 'Bicycle']
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)] # Blue, Green, Red

    for img_path in selected_images:
        # --- Create Prediction Image ---
        results = model(img_path)
        img_pred = cv2.imread(img_path)
        h, w, _ = img_pred.shape

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0]
                cls = int(box.cls[0])
                label = f'{class_names[cls]} {conf:.2f}'
                color = colors[cls]
                cv2.rectangle(img_pred, (x1, y1), (x2, y2), color, 2)
                cv2.putText(img_pred, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # --- Create Ground Truth Image ---
        img_gt = cv2.imread(img_path)
        # Handle different possible extensions for the label file
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(val_label_dir, f'{base_name}.txt')
        
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    cls = int(parts[0])
                    x_center, y_center, box_w, box_h = map(float, parts[1:])
                    x1 = int((x_center - box_w / 2) * w)
                    y1 = int((y_center - box_h / 2) * h)
                    x2 = int((x_center + box_w / 2) * w)
                    y2 = int((y_center + box_h / 2) * h)
                    color = colors[cls]
                    label = class_names[cls]
                    cv2.rectangle(img_gt, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(img_gt, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # --- Display Side-by-Side ---
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 9))
        ax1.imshow(cv2.cvtColor(img_gt, cv2.COLOR_BGR2RGB))
        ax1.set_title('Ground Truth')
        ax1.axis('off')

        ax2.imshow(cv2.cvtColor(img_pred, cv2.COLOR_BGR2RGB))
        ax2.set_title('YOLO-AIR Prediction')
        ax2.axis('off')

        plt.suptitle(f'Image: {os.path.basename(img_path)}', fontsize=16)
        plt.show()

    print("-" * 50)
    print(" Visualization complete!")
