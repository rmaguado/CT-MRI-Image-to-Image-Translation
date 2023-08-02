"""
This script is used to split the TCIA dataset into train, test, and validation.
Slices from the same patient are spread out into different batches.
"""

import os
import sys
import numpy as np
import nibabel as nib
import pandas as pd
import random
from tqdm import tqdm
import cv2

SAVE_ROOT = "/nfs/home/clruben/workspace/nst/data/"
BATCH_SIZE = 16
IMG_SIZE = 512
DATASET_RATIOS = {
    "train" : 0.90,
    "test" : 0.05,
    "val" : 0.05
}

def count_total_slices(sources, mode):
    total_slices = 0
    fails = 0
    fail_paths = []
    loop = tqdm(range(len(sources)))
    for i in loop:
        try:
            filepath = os.path.join(
                sources.iloc[i]["FinalPath"],
                "image.nii.gz"
            )
            nii_ = nib.load(filepath)
            image = nii_.get_fdata()
        except:
            fails += 1
            fail_paths.append(sources.iloc[i]["FinalPath"])
            continue
        if mode == "MR":
            if len(image.shape) > 3 and np.min(image) < 0:
                continue
        total_slices += min(image.shape)
    if fails > 0:
        print(f"Failed to load {fails} nifti files")
        print("\n".join(fail_paths))
    return total_slices

def winsorize_and_rescale(image, limits):
    lower_limit, upper_limit = limits
    x = np.where(image < lower_limit, lower_limit, image)
    x = np.where(x > upper_limit, upper_limit, x)
    x = (x - lower_limit) / (upper_limit - lower_limit)
    return x

def resize(x):
    num_slices = min(x.shape)
    slice_dim = x.shape.index(num_slices)
    big_size = max([i for i in x.shape if i is not num_slices])
    big_dim = [i for i, j in enumerate(x.shape) if j == big_size and i != slice_dim][0]
    small_dim = [i for i in [0,1,2] if i not in [big_dim, slice_dim]][0]
    small_size = x.shape[small_dim]
    x = np.transpose(x, (big_dim, small_dim, slice_dim))
    if big_size > small_size:
        pad_left = int((big_size - small_size)/2)
        pad_right = big_size - small_size - pad_left
        x = np.pad(x, [[0, 0], [pad_left, pad_right], [0, 0]])

    if big_size != IMG_SIZE:
        res_size = (
            IMG_SIZE,
            IMG_SIZE
        )
        x = cv2.resize(x, dsize=res_size, interpolation=cv2.INTER_CUBIC)
    x = np.transpose(x, (2, 0, 1))
    x = np.reshape(x, (num_slices, 1, IMG_SIZE, IMG_SIZE))
    return x

def create_mmap(dataset, mode, shape):
    global SAVE_ROOT
    filename = "_".join([str(x) for x in shape]) + ".npy"
    save_path = os.path.join(SAVE_ROOT, dataset, mode, filename)
    
    memmap_array = np.memmap(
        save_path, 
        dtype=np.float32, 
        mode='w+', 
        shape=shape
    )
    return memmap_array

def main(dataset_root, mode="CT"):

    sources = pd.read_csv(dataset_root)[:10]

    total_slices = count_total_slices(sources, mode)
    print(f"Total slices: {total_slices}")


    total_batches = total_slices//BATCH_SIZE

    train_batches = int(total_batches*DATASET_RATIOS["train"])
    test_batches = int(total_batches*DATASET_RATIOS["test"])
    val_batches = int(total_batches*DATASET_RATIOS["val"])

    if mode == "CT":
        LIMITS = (-1024, 3071)
    else:
        LIMITS = (0, 2000)

    train_mmap = create_mmap(
        "train",
        mode,
        (train_batches, BATCH_SIZE, 1, IMG_SIZE, IMG_SIZE)
    )
    test_mmap = create_mmap(
        "test",
        mode,
        (test_batches, BATCH_SIZE, 1, IMG_SIZE, IMG_SIZE)
    )
    val_mmap = create_mmap(
        "val",
        mode,
        (val_batches, BATCH_SIZE, 1, IMG_SIZE, IMG_SIZE)
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
        filename = os.path.join(
            sources.iloc[i]["FinalPath"],
            "image.nii.gz"
        )
        nii_ = nib.load(filename)
        try:
            nii_data = nii_.get_fdata().astype(np.float32)
        except:continue
        if mode == "MR" and np.min(nii_data) < 0:
            continue
        if len(nii_data.shape) > 3:
            continue
        nii_data = resize(nii_data)
        nii_data = winsorize_and_rescale(nii_data, LIMITS)

        for slice_ in nii_data:
            if train_counter < train_batches:
                train_mmap[train_counter][batch_counter] = slice_
                train_counter += 1
                continue
            if test_counter < test_batches:
                test_mmap[test_counter][batch_counter] = slice_
                test_counter += 1
                continue
            if val_counter < val_batches:
                val_mmap[val_counter][batch_counter] = slice_
                val_counter += 1
                continue
            train_counter = 0
            test_counter = 0
            val_counter = 0
            batch_counter += 1
            if batch_counter == BATCH_SIZE:
                break

# python3 /nfs/rwork/DATABASES_OPENSOURCE/TCIA_anon/CT_selection.csv CT
if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
