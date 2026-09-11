```
sudo apt install python3.10 python3.10-venv python3.10-dev



mkdir spurrin
cd spurrin
python3.10 -m venv nnunetv2_env
source nnunetv2_env/bin/activate
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install tqdm

mkdir nnUNet_setup
cd nnUNet_setup
git clone https://github.com/MIC-DKFZ/nnUNet.git
cd nnUNet
pip install -e .
# hiddenlayer enables nnU-net to generate plots of the network topologies it generates:
pip install --upgrade git+https://github.com/FabianIsensee/hiddenlayer.git    
pip install nnunetv2

cd ..
mkdir nnUNetFrame
cd nnUNetFrame
mkdir dataset
cd dataset
mkdir nnUNet_raw
mkdir nnUNet_preprocessed
mkdir nnUNet_results

<!-- /home/ubuntu/spurrin/nnUNet_setup/nnUNetFrame/dataset -->

nano ~/.bashrc

export nnUNet_raw=/home/ubuntu/spurrin/nnUNet_setup/nnUNetFrame/dataset/nnUNet_raw
export nnUNet_preprocessed=/home/ubuntu/spurrin/nnUNet_setup/nnUNetFrame/dataset/nnUNet_preprocessed
export nnUNet_results=/home/ubuntu/spurrin/nnUNet_setup/nnUNetFrame/dataset/nnUNet_results

source ~/.bashrc
echo ${nnUNet_raw}

```


```
- Run the format_naming_nnunet.py

/home/ubuntu/spurrin/nnUNet_setup/nnUNetFrame/dataset/nnUNet_raw/Dataset101_BHSD/dataset.json

{
  "name": "BleedBrain",
  "description": "Multi-class brain hemorrhage segmentation on non-contrast CT",
  "tensorImageSize": "3D",
  "reference": "BHSD",
  "licence": "research-only",
  "channel_names": {
    "0": "CT"
  },
  "labels": {
    "background": 0,
    "epidural": 1,
    "intraparenchymal": 2,
    "intraventricular": 3,
    "subarachnoid": 4,
    "subdural": 5
  },
  "numTraining": 182,
  "file_ending": ".nii.gz"
}


nnUNetv2_plan_and_preprocess -d 001 -c 3d_fullres --verify_dataset_integrity

nnUNetv2_plan_and_preprocess -d 101 -c 3d_fullres -pl nnUNetPlannerResEncM --verify_dataset_integrity


python -m nnunetv2.run.run_training \
  Dataset101_BHSD \
  3d_fullres \
  0 \
  -p nnUNetResEncUNetMPlans
```



# critical mismatches:
```
python - <<'EOF'
import os, glob
import nibabel as nib
import numpy as np

base = "/home/ubuntu/spurrin/nnUNet_setup/nnUNetFrame/dataset/nnUNet_raw/Dataset101_BHSD"
imgs = sorted(glob.glob(os.path.join(base, "imagesTr", "*_0000.nii.gz")))
bad = 0

def is_zero(m): 
    return np.allclose(m, 0)

for ip in imgs:
    case = os.path.basename(ip).replace("_0000.nii.gz", "")
    lp = os.path.join(base, "labelsTr", case + ".nii.gz")
    if not os.path.exists(lp):
        print("MISSING LABEL:", case)
        bad += 1
        continue

    I = nib.load(ip); L = nib.load(lp)

    # shape + spacing check
    if I.shape != L.shape:
        print("SHAPE MISMATCH:", case, I.shape, L.shape); bad += 1

    if tuple(np.round(I.header.get_zooms(), 6)) != tuple(np.round(L.header.get_zooms(), 6)):
        print("SPACING MISMATCH:", case, I.header.get_zooms(), L.header.get_zooms()); bad += 1

    # sform/qform sanity (warn-only)
    if is_zero(L.get_sform()):
        print("LABEL SFORM ZERO:", case)

print("\nDone. Critical mismatches:", bad)
EOF

```


# renaming the val images according to nnunet naming convention
```
cd /home/ubuntu/spurrin/nnUNet_setup/nnUNetFrame/dataset/label_192/val_images

for f in *.nii.gz; do
  base="${f%.nii.gz}"
  # already correct
  if [[ "$base" == *_0000 ]]; then
    continue
  fi
  mv "$f" "${base}_0000.nii.gz"
done


# reanming val labels
cd /home/ubuntu/spurrin/nnUNet_setup/nnUNetFrame/dataset/label_192/val_ground_truths

for f in *.nii.gz; do
  base="${f%.nii.gz}"
  # if label mistakenly has _0000, strip it
  if [[ "$base" == *_0000 ]]; then
    mv "$f" "${base%_0000}.nii.gz"
  fi
done

# match labels and images
comm -3 \
  <(ls /home/ubuntu/spurrin/nnUNet_setup/nnUNetFrame/dataset/label_192/val_images | sed 's/_0000\.nii\.gz$/.nii.gz/' | sort) \
  <(ls /home/ubuntu/spurrin/nnUNet_setup/nnUNetFrame/dataset/label_192/val_ground_truths | sort)

- if it prints nothing that means its correct

```

# run inference
```
nnUNetv2_predict \
  -i /home/ubuntu/spurrin/nnUNet_setup/nnUNetFrame/dataset/label_192/val_images \
  -o /home/ubuntu/spurrin/nnUNet_setup/nnUNetFrame/dataset/nnUNet_predictions/Dataset101_fold0_val_images \
  -d 101 \
  -c 3d_fullres \
  -f 0 \
  -p nnUNetResEncUNetMPlans \
  -chk checkpoint_best.pth

```

# run evaluation
```
nnUNetv2_evaluate_folder \
  -djfile /home/ubuntu/spurrin/nnUNet_setup/nnUNetFrame/dataset/nnUNet_raw/Dataset101_BHSD/dataset.json \
  -pfile /home/ubuntu/spurrin/nnUNet_setup/nnUNetFrame/dataset/nnUNet_predictions/Dataset101_fold0_val_images/plans.json \
  /home/ubuntu/spurrin/nnUNet_setup/nnUNetFrame/dataset/label_192/val_ground_truths \
  /home/ubuntu/spurrin/nnUNet_setup/nnUNetFrame/dataset/nnUNet_predictions/Dataset101_fold0_val_images

# class wise evaluation score

python - <<'EOF'
import json

path = "/home/ubuntu/spurrin/nnUNet_setup/nnUNetFrame/dataset/nnUNet_predictions/Dataset101_fold0_val_images/summary.json"
with open(path) as f:
    s = json.load(f)

print("\nClass-wise Dice:")
for label_name, metrics in s["mean"].items():
    if label_name.lower() != "background":
        print(f"{label_name:>15} : Dice = {metrics['Dice']:.4f}")
EOF

```


# basics commands
```
- cd /home/ubuntu/spurrin/nnUNet_setup/
- zip -r nnUNetFrame_v0_3d_BHSD.zip nnUNetFrame/
- rsync --progress -e "ssh -i neoscan_access.pem" ubuntu@13.126.111.28:/home/ubuntu/spurrin/nnUNet_setup_v0.zip .
```

---

# Repository overview

This repo trains and evaluates a 3D **nnU-Net v2** model for multi-class brain
hemorrhage segmentation (dataset: **BHSD**, `Dataset101_BHSD`) on non-contrast
head CT. It contains the nnU-Net v2 framework source, the project's dataset
working area (raw/preprocessed/results), and a small set of helper scripts
used to prepare data and evaluate predictions.

## Top-level layout

```
.
├── README.md                 # this file - setup, training/inference/eval commands
├── metrics.md                 # per-class Dice/IoU results + comparison vs BHSD paper baseline
├── .gitignore                  # ignores nnU-Net data dirs, model weights, imaging files, archives
├── nnunet_folder_Structure.png # reference screenshot of the expected folder layout
├── nnUNet/                    # vendored nnU-Net v2 framework (cloned from MIC-DKFZ/nnUNet)
├── nnUNetFrame/                # project data workspace (raw / preprocessed / results / predictions)
└── scripts/                    # project-specific helper scripts (not part of nnU-Net itself)
```

## `nnUNet/` — framework source

A local checkout of the [MIC-DKFZ/nnUNet](https://github.com/MIC-DKFZ/nnUNet)
v2 framework, installed editable (`pip install -e .`) as described in the
setup commands above. Notable subpaths:

- `nnunetv2/` — the installed Python package:
  - `experiment_planning/` — dataset fingerprinting and plan generation
    (`nnUNetv2_plan_and_preprocess`, `nnUNetPlannerResEncM`, etc.)
  - `preprocessing/` — cropping, resampling, normalization
  - `training/nnUNetTrainer/` — trainer classes used by `run_training`
  - `training/dataloading/`, `training/data_augmentation/`, `training/loss/`,
    `training/lr_scheduler/` — training pipeline components
  - `inference/` — `predict_from_raw_data.py`, sliding-window prediction,
    export of predictions (backs the `nnUNetv2_predict` CLI)
  - `evaluation/` — `evaluate_predictions.py`, `find_best_configuration.py`
    (backs the `nnUNetv2_evaluate_folder` CLI)
  - `postprocessing/`, `ensembling/`, `model_sharing/` — connected component
    cleanup, multi-model ensembling, exporting/importing trained models
  - `utilities/`, `imageio/`, `dataset_conversion/` — I/O readers/writers
    (NIfTI/SimpleITK/TIFF/natural images), plans/label handling, and example
    `DatasetXXX_*.py` conversion scripts for public datasets (BraTS, KiTS,
    AMOS, AutoPET, ToothFairy2, etc.)
  - `batch_running/`, `run/` — cluster batch-run helpers and the
    `run_training.py` entrypoint used in the training command above
  - `tests/` — unit and integration tests shipped with nnU-Net
- `documentation/` — upstream nnU-Net docs (dataset format, pretraining/
  fine-tuning, region-based training, environment variables, migration guide
  from v1, etc.) plus `assets/` images and `competitions/` write-ups
- `pyproject.toml`, `setup.py`, `nnunetv2.egg-info/` — packaging metadata for
  the editable install

## `nnUNetFrame/` — dataset workspace

Working directory pointed to by the `nnUNet_raw` / `nnUNet_preprocessed` /
`nnUNet_results` environment variables set up in the install steps above.
**All contents under `nnUNetFrame/dataset/` are data/weights and are excluded
from git** via `.gitignore` (see below) — this section documents the layout
for local reproduction, not tracked files.

```
nnUNetFrame/dataset/
├── nnUNet_raw/
│   └── Dataset101_BHSD/
│       ├── dataset.json      # channel names, label map, numTraining, file_ending
│       ├── imagesTr/         # 182 training CT volumes (*_0000.nii.gz)
│       ├── labelsTr/         # 182 matching voxelwise label maps (*.nii.gz)
│       └── imagesTs/         # held-out test images (unlabeled)
├── nnUNet_preprocessed/
│   └── Dataset101_BHSD/
│       ├── gt_segmentations/       # resampled ground-truth segmentations
│       └── nnUNetPlans_3d_fullres/ # preprocessed 3d_fullres training data
├── nnUNet_results/
│   └── Dataset101_BHSD/
│       └── nnUNetTrainer__nnUNetResEncUNetMPlans__3d_fullres/
│           ├── plans.json, dataset.json, dataset_fingerprint.json
│           └── fold_0/
│               ├── checkpoint_best.pth / checkpoint_final.pth  # model weights (ignored)
│               ├── progress.png, debug.json, training_log_*.txt
│               └── validation/     # per-case predictions + summary.json for fold 0
├── nnUNet_results.zip          # zipped copy of nnUNet_results for transfer (ignored)
├── nnUNet_predictions/
│   └── Dataset101_fold0_val_images/  # inference output on the held-out val set
└── label_192/                  # a 192-case labeled subset used for train/val splits
    ├── images/, ground_truths/                     # full labeled pool
    ├── val_images/, val_ground_truths/              # nnU-Net-naming-convention val split
    └── original_val_images/, original_val_ground_truths/  # pre-rename originals
```

## `scripts/` — project helper scripts

Standalone scripts (independent of the nnU-Net package) used to prepare and
sanity-check data for this project:

- `format_naming_nnunet.py` — converts the BHSD labeled subset (images +
  voxelwise masks) into the nnU-Net v2 raw-dataset layout, i.e. copies/renames
  volumes into `imagesTr/*_0000.nii.gz` and labels into `labelsTr/*.nii.gz`,
  matching each image to its label by filename stem.
- `test_preprocessed_data.py` — loads every `*.nii.gz` segmentation under a
  given directory (e.g. `nnUNet_preprocessed/.../gt_segmentations`) and prints
  the unique label values and nonzero voxel counts per case, to spot-check
  that labels survived preprocessing correctly.
- `test_labels.py` — loads a single predicted `*.nii.gz` volume and prints its
  unique values and nonzero voxel count, for quick manual inspection of one
  prediction.

## `metrics.md` — evaluation results

Per-class and aggregate Dice/IoU metrics for the trained
`nnUNetResEncUNetMPlans` / `3d_fullres` model on the BHSD validation set (5
hemorrhage subtypes: EDH, IPH, IVH, SAH, SDH), plus a side-by-side comparison
against the Dice scores reported in the original BHSD paper's nnUNet3D
baseline, and a discussion of where the model over/under-performs the
benchmark (see the file for full numbers and commentary).

## `.gitignore`

Keeps the repo to source/config/docs only. It excludes:
- OS/editor cruft (`.DS_Store`, `.vscode/`, `.idea/`) and Python build
  artifacts (`__pycache__/`, `*.egg-info/`, caches, logs)
- The bulky, regenerable nnU-Net data directories under `nnUNetFrame/dataset/`
  (`nnUNet_raw`, `nnUNet_preprocessed`, `nnUNet_predictions`, `label_192`,
  and the `validation/` subfolders + zip of `nnUNet_results`)
- Model weight/checkpoint files anywhere in the repo (`*.pth`, `*.pt`,
  `*.ckpt`, `*.safetensors`, etc.)
- Medical imaging and array formats anywhere in the repo (`*.nii.gz`, `*.dcm`,
  `*.npy`, `*.nrrd`, etc.)
- Archives (`*.zip`, `*.tar.gz`, etc.)