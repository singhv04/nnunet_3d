import glob, nibabel as nib, numpy as np

paths = sorted(glob.glob(
#   "/home/singhv04/Documents/spurrin/nnUNetFrame/dataset/nnUNet_preprocessed/Dataset101_BHSD/gt_segmentations/*.nii.gz"
  # "/home/singhv04/Documents/spurrin/nnUNetFrame/dataset/nnUNet_raw/Dataset101_BHSD/labelsTr/*.nii.gz"
  "/home/ubuntu/spurrin/nnUNet_setup/nnUNetFrame/dataset/nnUNet_preprocessed/Dataset101_BHSD/gt_segmentations/*.nii.gz"

))

for p in paths:
    seg = nib.load(p).get_fdata()
    u = np.unique(seg)
    # show counts for non-zero labels
    nz = seg[seg > 0]
    print(p.split("/")[-1], "unique:", u[:20], "…", "nonzero_voxels:", nz.size)
