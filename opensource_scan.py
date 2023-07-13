import os
import gzip
import numpy as np
from glob import glob
import nibabel as nib

BATCH_SIZE = 10
TEST_PERCENT = 5
VAL_PERCENT = 5
TRAIN_PERCENT = 100 - TEST_PERCENT - VAL_PERCENT
img_size = 512

def winsorize_and_rescale(image):
    x = np.where(image < -1024, -1024, image)
    x = np.where(x > 3071, 3071, x)
    x = (x + 1024) / 4095
    return x

def save_gzip(path, data):
    f = gzip.GzipFile(path, "w")
    np.save(f, data)
    f.close()

dataset_root = "../../../../rwork/DATABASES_OPENSOURCE/TCIA_anon/"
save_root = "/nfs/home/clruben/workspace/nst/data/preprocessed"

#datasets = next(os.walk(dataset_root))[1]
datasets = ["CMB-LCA"]
mode = "CT"
nifti_data = []
for dataset in datasets:

    dataset_dir = os.path.join(dataset_root, dataset)

    patient_dirs = next(os.walk(dataset_dir))[1]
    ct_dirs = []
    for patient_dir in patient_dirs:
        ct_dirs += glob(
            os.path.join(dataset_dir, patient_dir, "*_" + mode)
        )
    nifti_dirs = []
    for ct_dir in ct_dirs:
        nifti_dirs += glob(
            os.path.join(ct_dir, "nrrd/*.nii.gz")
        )
    for nifti_dir in nifti_dirs:
        nii_ = nib.load(nifti_dir)
        nii_data = nii_.get_fdata()
        nifti_data.append(
            nii_data
        )

np_nifti_data = np.concatenate(nifti_data, axis=2)
np_nifti_data = np.transpose(np_nifti_data, axes=(2,0,1))

rng = np.random.default_rng()
rng.shuffle(np_nifti_data, axis=0)

print(np_nifti_data.shape)

total_slices = len(np_nifti_data)
total_batches = total_slices // BATCH_SIZE

train_num_samples = int(total_slices*TRAIN_PERCENT/100) // BATCH_SIZE * BATCH_SIZE
test_num_samples = int(total_slices*TEST_PERCENT/100) // BATCH_SIZE * BATCH_SIZE
val_num_samples = int(total_slices*VAL_PERCENT/100) // BATCH_SIZE * BATCH_SIZE

test_delimiter = train_num_samples+test_num_samples
val_delimiter = train_num_samples+test_num_samples+val_num_samples

train_slices = np_nifti_data[:train_num_samples,:,:]
test_slices = np_nifti_data[train_num_samples:test_delimiter,:,:]
val_slices = np_nifti_data[test_delimiter:val_delimiter,:,:]

train_batch_data = np.reshape(train_slices, (train_num_samples//BATCH_SIZE, BATCH_SIZE, 1, img_size, img_size))
test_batch_data = np.reshape(test_slices, (test_num_samples//BATCH_SIZE, BATCH_SIZE, 1, img_size, img_size))
val_batch_data = np.reshape(val_slices, (val_num_samples//BATCH_SIZE, BATCH_SIZE, 1, img_size, img_size))

print(train_batch_data.shape)
print(test_batch_data.shape)
print(val_batch_data.shape)

print("train")
save_gzip(
    os.path.join(save_root, "train/CT.npy.gz"),
    train_batch_data
)
print("test")
save_gzip(
    os.path.join(save_root, "test/CT.npy.gz"),
    test_batch_data
)
print("val")
save_gzip(
    os.path.join(save_root, "val/CT.npy.gz"),
    val_batch_data
)
print("done")
