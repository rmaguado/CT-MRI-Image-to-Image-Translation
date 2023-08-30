"""
This script is used to split the TCIA dataset into train, test, and validation.
Slices from the same patient are spread out into different batches.
"""

import os
import sys
import numpy as np
import nibabel as nib
import pandas as pd
from tqdm import tqdm
import cv2

SAVE_ROOT = "/nfs/home/clruben/workspace/nst/data/"
IMG_SIZE = 512
DATASET_RATIOS = {
    "train" : 0.98,
    "test" : 0.02
}

def count_total_slices(sources, mode):
    total_slices = 0
    fails = 0
    fail_idx = []
    loop = tqdm(range(len(sources)))
    for i in loop:
        try:
            filepath = sources.iloc[i]["FinalPath"]
            nii_ = nib.load(filepath)
            image = nii_.get_fdata()
            resized = resize(image)
            if np.isnan(np.sum(resized)):
                fails += 1
                fail_idx.append(i)
                continue
        except Exception:
            fails += 1
            fail_idx.append(i)
            continue
        if mode == "MR":
            if len(image.shape) > 3 or np.min(image) < 0:
                fails += 1
                fail_idx.append(i)
                continue
        if np.isnan(np.sum(image)):
            fails += 1
            fail_idx.append(i)
            continue
        total_slices += min(image.shape)
    if fails > 0:
        print(f"Failed to load {fails} nifti files")
    # remove fails from sources
    sources = sources.drop(fail_idx)
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

    sources = pd.read_csv(dataset_root)

    total_slices = count_total_slices(sources, mode)
    print(f"Total slices: {total_slices}")

    train_slices = int(total_slices*DATASET_RATIOS["train"])
    test_slices = int(total_slices*DATASET_RATIOS["test"])

    if mode == "CT":
        limits = (-1024, 3071)
    else:
        limits = (0, 2000)

    train_mmap = create_mmap(
        "train",
        mode,
        (train_slices, 1, IMG_SIZE, IMG_SIZE)
    )
    test_mmap = create_mmap(
        "test",
        mode,
        (test_slices, 1, IMG_SIZE, IMG_SIZE)
    )

    dataset_indexes = list(range(len(sources)))
    #random.shuffle(dataset_indexes)
    loop = tqdm(dataset_indexes)

    train_counter = 0
    test_counter = 0

    for i in loop:
        filename = sources.iloc[i]["FinalPath"]
        try:
            nii_ = nib.load(filename)
            nii_data = nii_.get_fdata().astype(np.float32)
        except Exception:
            continue
        if mode == "MR":
            if len(nii_data.shape) > 3 or np.min(nii_data) < 0:
                continue
        nii_data = resize(nii_data)
        if np.isnan(np.sum(nii_data)):
            continue
        nii_data = winsorize_and_rescale(nii_data, limits)

        for slice_ in nii_data:
            if train_counter < train_slices:
                train_mmap[train_counter] = slice_
                train_counter += 1
                continue
            if test_counter < test_slices:
                test_mmap[test_counter] = slice_
                test_counter += 1
                continue
            break

# python3 split.py /nfs/rwork/DATABASES_OPENSOURCE/TCIA_anon/MR_selection.csv MR
if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
