"""
Contains a dataloader for the cyclegan model.
"""

import torch

from dataloaders import Dataset


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
        next_item = {
            "A": torch.from_numpy(next(self.loaders[0])),
            "B": torch.from_numpy(next(self.loaders[1]))
        }
        return next_item
