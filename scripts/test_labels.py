import nibabel as nib
import numpy as np

base_path = "/home/singhv04/Documents/spurrin/nnUNetFrame/dataset/predictions/test_d102/"
p = nib.load(base_path+"ID_0c6b97a9_ID_6011bb9ce8.nii.gz").get_fdata()
print(np.unique(p), np.count_nonzero(p))
