import os
import shutil
import random

# =======================
# CONFIGURATION
# =======================
SOURCE_DIR = "PlantVillage"   # folder containing class folders
DEST_DIR = "data"
TRAIN_RATIO = 0.8

# =======================
# CREATE TRAIN & TEST DIRS
# =======================
train_dir = os.path.join(DEST_DIR, "train")
test_dir = os.path.join(DEST_DIR, "test")

os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# =======================
# SPLIT DATA
# =======================
for class_name in os.listdir(SOURCE_DIR):
    class_path = os.path.join(SOURCE_DIR, class_name)

    if not os.path.isdir(class_path):
        continue

    images = os.listdir(class_path)
    random.shuffle(images)

    split_index = int(len(images) * TRAIN_RATIO)

    train_images = images[:split_index]
    test_images = images[split_index:]

    # Create class folders
    os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
    os.makedirs(os.path.join(test_dir, class_name), exist_ok=True)

    # Move images
    for img in train_images:
        shutil.copy(
            os.path.join(class_path, img),
            os.path.join(train_dir, class_name, img)
        )

    for img in test_images:
        shutil.copy(
            os.path.join(class_path, img),
            os.path.join(test_dir, class_name, img)
        )

    print(f"✅ {class_name}: {len(train_images)} train | {len(test_images)} test")

print("\n🎉 Dataset split completed successfully!")
