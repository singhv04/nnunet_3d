"""
Convert BHSD labeled subset (images + voxel-wise masks) into nnUNetv2 "raw dataset" format.

nnUNetv2 expects a dataset directory under nnUNet_raw like:

nnUNet_raw/
  Dataset001_BHSD/
    imagesTr/   -> training images (all labeled cases go here)
      CASEID_0000.nii.gz   # _0000 = channel/modality index (CT is single-modality -> channel 0)
    labelsTr/   -> training labels (same CASEID, no _0000)
      CASEID.nii.gz
    imagesTs/   -> optional test images WITHOUT labels (can be left empty)
    dataset.json -> dataset metadata (create separately)

This script:
- copies and renames image volumes to imagesTr/*_0000.nii.gz
- copies and renames corresponding labels to labelsTr/*.nii.gz
- matches images to labels by filename stem (without .nii.gz)
"""

import os
import shutil
from pathlib import Path
from tqdm import tqdm

# =========================
# USER CONFIGURATION
# =========================

# Base folder that contains your BHSD labeled subset directories.
# Expected structure:
#   <base_data_dir_path>/
#     images/          (CT volumes as .nii.gz)
#     ground_truths/   (segmentation masks as .nii.gz)

# base_data_dir_path = Path("/home/singhv04/Documents/spurrin/brain-ct/heamorrage/BHSD/trial")
base_data_dir_path = Path("/home/ubuntu/spurrin/nnUNet_setup/nnUNetFrame/dataset/label_192")

# Source folders
SRC_IMAGES = base_data_dir_path / "images"
SRC_LABELS = base_data_dir_path / "ground_truths"

# nnUNet raw base directory:
# nnUNetv2 uses env var nnUNet_raw if set; otherwise defaults to ~/nnUNet_raw
NNUNET_RAW = Path(os.environ.get("nnUNet_raw", str(Path.home() / "nnUNet_raw")))

# Output dataset folder name must follow: DatasetXXX_NAME
# - XXX is an integer ID used in nnUNet commands like: nnUNetv2_plan_and_preprocess -d 1
# - NAME is just a human-readable identifier
OUT_DS = NNUNET_RAW / "Dataset101_BHSD"

# If you want to make a held-out test set, you can copy some cases into imagesTs manually
# (no labels go there). Otherwise keep imagesTs empty.
# =========================


# =========================
# OUTPUT FOLDERS (nnUNet format)
# =========================
IMAGES_TR = OUT_DS / "imagesTr"
LABELS_TR = OUT_DS / "labelsTr"
IMAGES_TS = OUT_DS / "imagesTs"  # optional, can remain empty

IMAGES_TR.mkdir(parents=True, exist_ok=True)
LABELS_TR.mkdir(parents=True, exist_ok=True)
IMAGES_TS.mkdir(parents=True, exist_ok=True)


def stem_nii_gz(p: Path) -> str:
    """
    Return the filename stem for .nii.gz files.
    Example:
        "BHSD_001.nii.gz" -> "BHSD_001"
    """
    name = p.name
    if name.endswith(".nii.gz"):
        return name[:-7]  # remove trailing ".nii.gz"
    return p.stem


# =========================
# VALIDATION: ensure input folders exist
# =========================
if not SRC_IMAGES.exists():
    raise FileNotFoundError(f"SRC_IMAGES folder not found: {SRC_IMAGES}")
if not SRC_LABELS.exists():
    raise FileNotFoundError(f"SRC_LABELS folder not found: {SRC_LABELS}")

# Gather label files into a lookup dictionary keyed by CASEID (stem)
# This assumes image and label filenames match (same stem) e.g.:
#   images/BHSD_001.nii.gz
#   ground_truths/BHSD_001.nii.gz
label_map = {stem_nii_gz(p): p for p in SRC_LABELS.glob("*.nii.gz")}

missing_labels = []   # images that have no matching label
missing_images = []   # labels that have no matching image (useful check)
copied = 0


# =========================
# COPY + RENAME
# =========================
for img in tqdm(sorted(SRC_IMAGES.glob("*.nii.gz"))):
    case_id = stem_nii_gz(img)

    # Check label exists for this image
    if case_id not in label_map:
        missing_labels.append(case_id)
        continue

    # nnUNet requires modality/channel suffix for images:
    # - _0000 means "channel 0" (single-modality CT)
    # Labels do NOT have channel suffix.
    out_img = IMAGES_TR / f"{case_id}_0000.nii.gz"
    out_lab = LABELS_TR / f"{case_id}.nii.gz"

    # copy2 preserves timestamps/metadata; safe for dataset conversion
    shutil.copy2(img, out_img)
    shutil.copy2(label_map[case_id], out_lab)
    copied += 1


# Optional: detect labels that don't have corresponding images (helps catch typos/mismatched naming)
print("Data integrity check in process")
image_stems = {stem_nii_gz(p) for p in SRC_IMAGES.glob("*.nii.gz")}
for lab_stem in tqdm(label_map.keys()):
    if lab_stem not in image_stems:
        missing_images.append(lab_stem)


# =========================
# SUMMARY
# =========================
print(f"✅ Copied {copied} labeled cases into nnUNet raw dataset: {OUT_DS}")

if missing_labels:
    print(f"⚠️ {len(missing_labels)} image(s) without matching label. First 20: {missing_labels[:20]}")

if missing_images:
    print(f"⚠️ {len(missing_images)} label(s) without matching image. First 20: {missing_images[:20]}")

print("\nNext steps:")
print("1) Create dataset.json inside the dataset folder.")
print("2) Run: nnUNetv2_plan_and_preprocess -d 1 --verify_dataset_integrity")

