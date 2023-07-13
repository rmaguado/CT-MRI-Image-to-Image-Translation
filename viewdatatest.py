from glob import glob
import nibabel as nib
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import tifffile

#patient_paths = glob("../../../../rwork/DATABASES_OPENSOURCE/total_segmentator/Totalsegmentator_dataset/s*")

"""
Abrir nifti comprimido (.nii.gz):

nii_image = nib.load(test_path)
nii_data = nii_image.get_fdata()
"""

def winsorize_and_rescale(image):
    x = np.where(image < -1024, -1024, image)
    x = np.where(x > 3071, 3071, x)
    x = (x + 1024) / 4095
    return np.expand_dims(x, axis=0)

tiff_paths = glob("/nfs/home/clruben/workspace/nst/tempdata/tiff_images/*.tif")

images = []

for img_path in tiff_paths:
    im = tifffile.imread(img_path)
    images.append( winsorize_and_rescale(im) )

# image width height (no slices)
np_images = np.array(images, dtype=np.float32)
print(np_images.shape)

np.savez("/nfs/home/clruben/workspace/nst/tempdata/preprocessed/test_batch.npz", np_images)
