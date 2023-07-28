import os
import numpy as np
from PIL import Image
import nibabel as nib
import pandas as pd
import random
from tqdm import tqdm
import cv2

dataset_root = "/nfs/rwork/DATABASES_OPENSOURCE/TCIA_anon/Selected_CTs.csv"
save_root = "/nfs/home/clruben/workspace/nst/data/"
mode = "CT"

sources = pd.read_csv(dataset_root)

total_slices = 0
loop = tqdm(range(len(sources)))
for i in loop:
    try:
        nii_ = nib.load(
            sources.iloc[i]["FinalPath"]
        )
        image = nii_.get_fdata()
    except:
        continue
    if mode == "MRI":
        if len(image.shape) > 3 and np.min(image) < 0:
            continue
    total_slices += min(image.shape)

print(f"Total slices: {total_slices}")

BATCH_SIZE : int = 16
total_batches = total_slices//BATCH_SIZE
IMG_SIZE : int = 512
DATASET_RATIOS = {
    "train" : 0.90,
    "test" : 0.05,
    "val" : 0.05
}

TRAIN_BATCHES = int(total_batches*DATASET_RATIOS["train"])
TEST_BATCHES = int(total_batches*DATASET_RATIOS["test"])
VAL_BATCHES = int(total_batches*DATASET_RATIOS["val"])

if mode == "CT":
    LOWER_LIMIT = -1024
    UPPER_LIMIT = 3071
else:
    LOWER_LIMIT = 0
    UPPER_LIMIT = 2000

def winsorize_and_rescale(image):
    x = np.where(image < LOWER_LIMIT, LOWER_LIMIT, image)
    x = np.where(x > UPPER_LIMIT, UPPER_LIMIT, x)
    x = (x - LOWER_LIMIT) / (UPPER_LIMIT - LOWER_LIMIT)
    return x

def resize(x):
    num_slices = min(x.shape)
    big_size = max(x.shape)
    slice_dim = x.shape.index(num_slices)
    big_dim = [i for i in [0,1,2] if i is not slice_dim][0]
    small_dim = [i for i in [0,1,2] if i not in [big_dim, slice_dim]][0]
    small_size = x.shape[small_dim]
    x = np.transpose(x, (big_dim, small_dim, slice_dim))
    if big_size > small_size:
        pad_left = int((big_size - small_size)/2)
        pad_right = big_size - small_size - pad_left
        x = np.pad(x, [[0, 0], [pad_left, pad_right], [0, 0]])

    if big_dim != IMG_SIZE:
        res_size = (
            IMG_SIZE,
            IMG_SIZE
        )
        x = cv2.resize(x, dsize=res_size, interpolation=cv2.INTER_CUBIC)
    x = np.transpose(x, (big_dim, small_dim, slice_dim))
    x = np.reshape(x, (num_slices, 1, IMG_SIZE, IMG_SIZE))
    return x

def create_mmap(dataset, mode, shape):
    filename = "_".join([str(x) for x in shape]) + ".npy"
    save_path = os.path.join(save_root, dataset, mode, filename)
    
    memmap_array = np.memmap(
        save_path, 
        dtype=np.float32, 
        mode='w+', 
        shape=shape
    )
    return memmap_array

train_mmap = create_mmap(
    "train",
    mode,
    (TRAIN_BATCHES, BATCH_SIZE, 1, IMG_SIZE, IMG_SIZE)
)
test_mmap = create_mmap(
    "test",
    mode,
    (TEST_BATCHES, BATCH_SIZE, 1, IMG_SIZE, IMG_SIZE)
)
val_mmap = create_mmap(
    "val",
    mode,
    (VAL_BATCHES, BATCH_SIZE, 1, IMG_SIZE, IMG_SIZE)
)

rng = np.random.default_rng()
dataset_indexes = list(range(len(sources)))
random.shuffle(dataset_indexes)
loop = tqdm(dataset_indexes)

batch_counter = 0
train_counter = 0
test_counter = 0
val_counter = 0

for i in loop:
    nii_ = nib.load(
        sources.iloc[i]["FinalPath"]
    )
    try:
        nii_data = nii_.get_fdata().astype(np.float32)
    except:continue
    if mode == "MRI" and np.min(nii_data) < 0:
        continue
    if len(nii_data.shape) > 3:
        continue
    nii_data = resize(nii_data)
    nii_data = winsorize_and_rescale(nii_data)

    #rng.shuffle(joined_slices, axis=0)
    for slice_ in nii_data:
        if train_counter < TRAIN_BATCHES:
            train_mmap[train_counter][batch_counter] = slice_
            train_counter += 1
            continue
        if test_counter < TEST_BATCHES:
            test_mmap[test_counter][batch_counter] = slice_
            test_counter += 1
            continue
        if val_counter < VAL_BATCHES:
            val_mmap[val_counter][batch_counter] = slice_
            val_counter += 1
            continue
        train_counter = 0
        test_counter = 0
        val_counter = 0
        batch_counter += 1
        if batch_counter == BATCH_SIZE:
            break
