import os
import cv2
import yaml
import numpy as np
from glob import glob
from sklearn.model_selection import train_test_split

# Paths
DATASET_PATH = "/kaggle/input/polypdb/PolypDB/PolypDB/PolypDB_center_wise"
OUTPUT_PATH = "/kaggle/working/yolo_dataset"

# Allowed image and mask extensions
image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")
mask_extensions = (".png", ".jpg", ".jpeg", ".bmp", ".tiff")

# Create necessary directories
for split in ["train", "val", "test"]:
    os.makedirs(f"{OUTPUT_PATH}/images/{split}", exist_ok=True)
    os.makedirs(f"{OUTPUT_PATH}/labels/{split}", exist_ok=True)

# Function to get bounding boxes from masks
def get_bounding_boxes(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        x_center, y_center = x + w / 2, y + h / 2
        boxes.append([x_center, y_center, w, h])
    return boxes

# Collect all valid image-mask pairs with labels
all_data = []
class_names = set()

for center in os.listdir(DATASET_PATH):
    center_path = os.path.join(DATASET_PATH, center)
    
    if not os.path.isdir(center_path):
        continue  # Skip non-directory files
    
    for modality in os.listdir(center_path):
        modality_path = os.path.join(center_path, modality)

        img_dir = os.path.join(modality_path, "images")
        mask_dir = os.path.join(modality_path, "masks")

        if not os.path.exists(img_dir) or not os.path.exists(mask_dir):
            continue  # Skip if either images or masks folder is missing

        for img_path in glob(f"{img_dir}/*"):
            if not img_path.endswith(image_extensions):
                continue  # Skip non-image files

            img_name = os.path.basename(img_path)
            img_base = os.path.splitext(img_name)[0]  # Filename without extension

            # Find corresponding mask
            mask_path = None
            for ext in mask_extensions:
                candidate_mask = os.path.join(mask_dir, img_base + ext)
                if os.path.exists(candidate_mask):
                    mask_path = candidate_mask
                    break

            if mask_path:
                label = f"{center}_{modality}"
                class_names.add(label)
                all_data.append((img_path, mask_path, label))
            else:
                print(f"⚠️ No mask found for {img_path}")

# Assign unique class IDs
class_names = sorted(class_names)  # Sort for consistency
class_to_id = {name: idx for idx, name in enumerate(class_names)}

# Split dataset (80% Train, 10% Val, 10% Test)
train_data, test_data = train_test_split(all_data, test_size=0.2, random_state=42)
val_data, test_data = train_test_split(test_data, test_size=0.5, random_state=42)

# Function to save images and labels in YOLO format
def save_data(split, data):
    for img_path, mask_path, label in data:
        img_name = os.path.basename(img_path)
        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if img is None or mask is None:
            continue  # Skip corrupted files

        bboxes = get_bounding_boxes(mask)
        if not bboxes:
            continue  # Skip images without bounding boxes

        # Save image
        new_img_path = f"{OUTPUT_PATH}/images/{split}/{img_name}"
        cv2.imwrite(new_img_path, img)

        # Save YOLO label file
        label_id = class_to_id[label]
        label_path = f"{OUTPUT_PATH}/labels/{split}/{img_name.replace('.jpg', '.txt')}"
        with open(label_path, "w") as f:
            for box in bboxes:
                x, y, w, h = box
                H, W = img.shape[:2]
                f.write(f"{label_id} {x/W} {y/H} {w/W} {h/H}\n")

# Save train, val, test sets
save_data("train", train_data)
save_data("val", val_data)
save_data("test", test_data)

# Generate data.yaml file dynamically
data_yaml = {
    "train": f"{OUTPUT_PATH}/images/train",
    "val": f"{OUTPUT_PATH}/images/val",
    "test": f"{OUTPUT_PATH}/images/test",
    "nc": len(class_names),
    "names": class_names
}

with open(f"{OUTPUT_PATH}/data.yaml", "w") as yaml_file:
    yaml.dump(data_yaml, yaml_file, default_flow_style=False)

print(f"✅ Dataset converted to YOLO format! Found {len(all_data)} valid image-mask pairs.")
print(f"📂 Classes: {class_names}")    
