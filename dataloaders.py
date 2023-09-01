"""Dataloaders for the NST project.
"""

from typing import Optional
from glob import glob
import os

import numpy as np
import torch


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
            batch_size: int = 4,
            enable_shuffle: bool = True
    ):
        dataset_path = os.path.join(data_root_path, dataset)

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
        self.data = np.memmap(
            data_dir,
            dtype=np.float32,
            mode='r',
            shape=data_shape
        )
        self.batch_size = batch_size
        self.counter = 0
        self.total_images = data_shape[0]
        self.index_list = np.arange(self.total_images)
        if enable_shuffle:
            np.random.shuffle(self.index_list)

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return self.total_images // self.batch_size

    def __iter__(self):
        return self

    def __next__(self):
        if self.counter == self.total_images:
            self.counter = 0
        batch = self.data[
            self.index_list[self.counter:self.counter+self.batch_size]
        ]
        self.counter += self.batch_size
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
            batch_size: int = 4
    ):
        self.loaders = [
            Dataset(data_root_path, dataset_name, "CT", batch_size=batch_size),
            Dataset(data_root_path, dataset_name, "MR", batch_size=batch_size)
        ]
        self.loop_length = min(
            [len(x) for x in self.loaders]
        ) * 2
        self.mode = True
        self.counter = 0
        
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

    def __len__(self):
        return self.loop_length

    def __iter__(self):
        return self

    def __next__(self):
        self.mode = not self.mode
        if self.counter == self.loop_length:
            self.counter = 0
        next_item = torch.from_numpy(
            next(self.loaders[self.mode])
        )
        mode = ["CT", "MR"][self.mode]
        return next_item, mode

class CycleganDataloader:
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
            batch_size: int = 4
    ):
        self.loaders = [
            Dataset(data_root_path, dataset_name, "CT", batch_size=batch_size),
            Dataset(data_root_path, dataset_name, "MR", batch_size=batch_size)
        ]
        self.loop_length = min(
            [len(x) for x in self.loaders]
        ) * 2
        self.counter = 0

    def __len__(self):
        return self.loop_length

    def __iter__(self):
        return self

    def __next__(self):
        if self.counter == self.loop_length:
            self.counter = 0
        return torch.from_numpy(next(self.loaders[0])), \
            torch.from_numpy(next(self.loaders[1]))

class UNetDataloader:
    """Obtaines paired CT and MR batches from separate datasets.

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
            size_limit: Optional[int] = None,
            enable_data_augmentation: bool = False
    ):
        self.loaders = [
            Dataset(data_root_path, dataset_name, "MR", enable_shuffle=False),
            Dataset(data_root_path, dataset_name, "CT", enable_shuffle=False)
        ]
        if size_limit is None:
            self.loop_length = min(
                [len(x) for x in self.loaders]
            ) * 2
        else:
            self.loop_length = size_limit
        self.counter = 0
        self.enable_data_augmentation = enable_data_augmentation
        
    def data_augmentation(self, batch_a, batch_b):
        """Reflects and rotates the images at random. Applies the same
        transformation to paired images.

        Args:
            batch_a (np.ndarray): Shape (B, C, H, W)
            batch_b (np.ndarray): Shape (B, C, H, W)
        Returns:
            (nd.array) : Transformed batch_data (B, C, H, W)
            (nd.array) : Transformed batch_data (B, C, H, W)
        """
        num_rotations = np.random.randint(4)
        batch_a = np.rot90(batch_a, num_rotations, axes=(2, 3))
        batch_b = np.rot90(batch_b, num_rotations, axes=(2, 3))

        if np.random.random() < 0.5:
            batch_a = np.flip(batch_a, axis=3)
            batch_b = np.flip(batch_b, axis=3)

        return batch_a, batch_b

    def __len__(self):
        return self.loop_length

    def __iter__(self):
        return self

    def __next__(self):
        if self.counter == self.loop_length:
            self.counter = 0
        batch_a = next(self.loaders[0])
        batch_b = next(self.loaders[1])
        if self.enable_data_augmentation:
            batch_a, batch_b = self.data_augmentation(batch_a, batch_b)
        # inputs, targets
        return batch_a, batch_b
