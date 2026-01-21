import os
import glob
import random
import json
import shutil
from collections import defaultdict


# 1️⃣ Collect and group CSVs by prefix
root_dir = r"c:\Users\Mark\Desktop\PROJECT 1\250225_인턴교육_마크\1주차\Dataset\3W Dataset\dataset"

csv_paths = glob.glob(root_dir + '/**/*.csv', recursive=True)
groups = defaultdict(list)

for path in csv_paths:
    fname = os.path.basename(path)
    if fname.startswith('WELL'):
        key = 'well'
    elif fname.startswith('SIMULATED'):
        key = 'simulated'
    elif fname.startswith('DRAWN'):
        key = 'drawn'
    else:
        raise ValueError(f"Unrecognized file prefix in {fname}")

    groups[key].append(path)

# Debug print: group stats and sample files
for key, paths in groups.items():
    print(f"\nGroup '{key}': {len(paths)} files")
    for p in paths[:3]:  # show first 3 paths
        print(f"  {p}")

# 2️⃣ Split each group into train/val/test
train_all, val_all, test_all = [], [], []
random.seed(42)

for key, paths in groups.items():
    random.shuffle(paths)
    n_total = len(paths)
    n_train = int(0.7 * n_total)
    n_val = int(0.15 * n_total)

    train = paths[:n_train]
    val = paths[n_train:n_train + n_val]
    test = paths[n_train + n_val:]

    print(f"{key}: train={len(train)}, val={len(val)}, test={len(test)}")

    train_all.extend(train)
    val_all.extend(val)
    test_all.extend(test)

# 3️⃣ Save splits as JSON lists of paths
with open('train_files.json', 'w') as f:
    json.dump(train_all, f)
with open('val_files.json', 'w') as f:
    json.dump(val_all, f)
with open('test_files.json', 'w') as f:
    json.dump(test_all, f)

print("\n Saved train_files.json, val_files.json, test_files.json")

# 4️⃣ Copy each split’s CSVs into separate folders
def copy_files(csv_paths, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    copied = 0
    for src in csv_paths:
        if not os.path.exists(src):
            print(f"WARNING: File not found -> {src}")
            continue
        dst = os.path.join(output_dir, os.path.basename(src))
        shutil.copy2(src, dst)
        copied += 1
    print(f"\n Copied {copied} files to {output_dir}")

copy_files(train_all, 'train_split')
copy_files(val_all, 'val_split')
copy_files(test_all, 'test_split')

print("\nDone! Train/val/test splits created and files copied into train_split/, val_split/, test_split/")







# import random

# random.seed(42)  # For reproducibility
# random.shuffle(csv_paths)

# n_total = len(csv_paths)
# n_train = int(0.7 * n_total)
# n_val = int(0.15 * n_total)

# train_paths = csv_paths[:n_train]
# val_paths = csv_paths[n_train:n_train + n_val]
# test_paths = csv_paths[n_train + n_val:]

# print(f"Train: {len(train_paths)}, Validation: {len(val_paths)}, Test: {len(test_paths)}")