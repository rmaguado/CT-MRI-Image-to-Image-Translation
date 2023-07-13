import os
import numpy as np
from glob import glob
import nibabel as nib

def winsorize_and_rescale(image):
    x = np.where(image < -1024, -1024, image)
    x = np.where(x > 3071, 3071, x)
    x = (x + 1024) / 4095
    return x

dataset_root = "../../../../rwork/DATABASES_OPENSOURCE/TCIA_anon/"

#datasets = next(os.walk(dataset_root))[1]
datasets = ["CMB-LCA"]

for dataset in datasets:

    dataset_dir = os.path.join(dataset_root, dataset)

    patient_dirs = next(os.walk(dataset_dir))[1]
    ct_dirs = []
    for patient_dir in patient_dirs:
        ct_dirs += glob(
            os.path.join(dataset_dir, patient_dir, "*_CT")
        )
    nifti_dirs = []
    for ct_dir in ct_dirs:
        nifti_dirs += glob(
            os.path.join(ct_dir, "nrrd/*.nii.gz")
        )
    nifti_data = []
    for nifti_dir in nifti_dirs:
        nii_ = nib.load(nifti_dir)
        nii_data = nii_.get_fdata()
        print(nii_data.shape)
        nifti_data.append(
            nii_data
        )
    np_nifti_data = np.concatenate(nifti_data, axis=2)
    print(np_nifti_data.shape)
