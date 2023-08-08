"""Dataloaders for the NST project.
"""

from typing import Optional
from glob import glob
import os

import numpy as np


class Dataset:
    """Loads a dataset from a .npy file.

    Args:
        data_root_path (str): Path to the directory containing the dataset
        dataset (str): Name of the dataset to load. e.g. "train" or "test"
        mode (str): Name of the dataset to load. e.g. "train" or "test"
        enable_data_augmentation (bool): Whether to enable data augmentation
    """
    def __init__(
            self,
            data_root_path: str,
            dataset: str = "train",
            mode: str = "CT",
            enable_data_augmentation: bool = False
    ):
        dataset_path = os.path.join(data_root_path, dataset)
        self.enable_data_augmentation = enable_data_augmentation

        paths_to_database = glob(
            os.path.join(dataset_path, mode, "*.npy")
        )
        if not paths_to_database:
            raise ValueError(
                "Data not found. Check source directory provided."
            )

        data_dir = paths_to_database[0]
        data_shape = tuple(
            [
                int(x) for x in os.path.basename(
                    data_dir
                ).replace(".npy", "").split("_")
            ]
        )
        self.batches = np.memmap(
            data_dir,
            dtype=np.float32,
            mode='r',
            shape=data_shape
        )
        self.counter = 0
        self.total_batches = data_shape[0]

    def data_augmentation(self, batch_data):
        """Reflects and rotates the image at random.

        Args:
            batch_data (np.ndarray): Shape (B, C, H, W)
        Returns:
            (nd.array) : Transformed batch_data (B, C, H, W)
        """
        num_rotations = np.random.randint(4)
        batch_data = np.rot90(batch_data, num_rotations, axes=(2, 3))

        if np.random.random() < 0.5:
            batch_data = np.flip(batch_data, axis=3)

        return batch_data

    def __getitem__(self, idx):
        return self.batches[idx]

    def __len__(self):
        return self.total_batches

    def __iter__(self):
        return self

    def __next__(self):
        if self.counter == self.total_batches:
            self.counter = 0
        batch = self.batches[self.counter]
        self.counter += 1
        if self.enable_data_augmentation:
            return self.data_augmentation(batch)
        return batch


class Dataloader:
    """Pools together MRI and CT scans for masked modeling.

    Assumes that data_root_path contains a directory called dataset, which
    contains two subdirectories, CT and MR, which contain .npy files. The names
    of the .npy files should contain the shape of the data, e.g.
    100_1_256_256.npy. The number of batches in each dataset does not need to
    be the same.

    Args:
        data_root_path (str): Path to the directory containing the dataset
        dataset_name (str): Name of the dataset to load. e.g. "train" or "test"
    """
    def __init__(
            self,
            data_root_path: str,
            dataset_name: str = "train",
            size_limit: Optional[int] = None
    ):
        self.loaders = [
            Dataset(data_root_path, dataset_name, "CT"),
            Dataset(data_root_path, dataset_name, "MR")
        ]
        if size_limit is None:
            self.loop_length = min(
                [len(x) for x in self.loaders]
            ) * 2
        else:
            self.loop_length = size_limit
        self.mode = True
        self.counter = 0

    def __len__(self):
        return self.loop_length

    def __iter__(self):
        return self

    def __next__(self):
        self.mode = not self.mode
        if self.counter == self.loop_length:
            self.counter = 0
        next_item = next(self.loaders[self.mode])
        mode = ["CT", "MR"][self.mode]
        return next_item, mode


if __name__ == "__main__":
    from tqdm import tqdm
    import time

    t0 = time.time()
    SOURCE = "/nfs/home/clruben/workspace/nst/data/preprocessed/"
    testloader = Dataset(SOURCE, "train", "CT")
    print(time.time()-t0)
    t0 = time.time()
    loop = tqdm(range(len(testloader)))
    for batch_data in loop:
        x = next(testloader)
    print(time.time()-t0)
