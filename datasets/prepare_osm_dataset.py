import os
import random
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser(description='Process image and label directories and generate train and validation files.')
parser.add_argument('--image_dir', default="osm_sentinel_imagery", help='Directory containing image .npz files')
parser.add_argument('--label_dir', default="osm_label_images_v10", help='Directory containing label files')
parser.add_argument('--output_dir', default=".", help='Directory to save the output files')
parser.add_argument('--val_split', type=float, default=0.1, help='Fraction of data to be used for validation (default: 0.1)')
parser.add_argument('--seed', type=int, default=None, help='Random seed for shuffling (default: None)')

args = parser.parse_args()

image_dir = args.image_dir
label_dir = args.label_dir
output_dir = args.output_dir
val_split = args.val_split
seed = args.seed

if seed is not None:
    random.seed(seed)

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Collect all .npz files, grouped by last subdirectory name
subdir_files = {}
for root, dirs, files in os.walk(image_dir):
    for file in files:
        if file.endswith('.npz'):
            # Generate absolute path for the file
            file_path = os.path.abspath(os.path.join(root, file))
            # Extract last subdirectory name based on the structure under image_dir
            subdir = os.path.basename(root)
            if subdir not in subdir_files:
                subdir_files[subdir] = {'images': [], 'labels': []}
            subdir_files[subdir]['images'].append(file_path)

# Collect labels matching the name of the last subdirectory containing the npz file
for root, dirs, files in os.walk(label_dir):
    for file in files:
        # Extract the subdirectory name from the filename without the extension
        subdir = file.split('.')[0]
        if subdir in subdir_files:
            label_path = os.path.abspath(os.path.join(root, file))
            subdir_files[subdir]['labels'].append(label_path)

# Determine train and val sets ensuring no last subdirectory overlaps
train_files = []
val_files = []
subdirs = list(subdir_files.keys())
random.shuffle(subdirs)  # Shuffle subdirectories to randomize allocation

# Allocate approximately val_split of the subdirectories to validation
val_count = int(val_split * len(subdirs))
val_subdirs = subdirs[:val_count]
train_subdirs = subdirs[val_count:]

# Assign files to train or val based on their subdirectory
for subdir in subdirs:
    entries = subdir_files[subdir]
    # Create pairs of each image with its corresponding labels
    paired_files = [(img, lbl) for img in entries['images'] for lbl in entries['labels']]
    if subdir in val_subdirs:
        val_files.extend(paired_files)
    else:
        train_files.extend(paired_files)

# Save the lists to files
with open(os.path.join(output_dir, 'train_files.txt'), 'w') as f:
    for img, lbl in train_files:
        f.write(f"{img}\t{lbl}\n")

with open(os.path.join(output_dir, 'val_files.txt'), 'w') as f:
    for img, lbl in val_files:
        f.write(f"{img}\t{lbl}\n")
