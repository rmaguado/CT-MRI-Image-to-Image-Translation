# access numpy files from /nfs/home/clruben/workspace/nst/data/reg/train/CT/3380_16_1_512_512.npy
# the array size is (3380, 16, 1, 512, 512)
# will create a memmap array of size (3380*16, 1, 512, 512) that is a view of the original array

import numpy as np
import os

data_root_path = "/nfs/home/clruben/workspace/nst/data/reg/"

ct_train_memmap = np.memmap(
    os.path.join(data_root_path, "train", "CT", "3380_16_1_512_512.npy"),
    dtype=np.float32,
    mode='r',
    shape=(3380, 16, 1, 512, 512)
)

reshaped_ct_train_memmap = np.memmap(
    os.path.join(data_root_path, "train", "CT", "54080_1_512_512_reshaped.npy"),
    dtype=np.float32,
    mode='w+',
    shape=(54080, 1, 512, 512)
)

for idx in range(3380):
    batched_data = ct_train_memmap[idx]
    for i in range(16):
        slice_ = batched_data[i]
        reshaped_ct_train_memmap[idx*16:(idx+1)*16] = slice_
