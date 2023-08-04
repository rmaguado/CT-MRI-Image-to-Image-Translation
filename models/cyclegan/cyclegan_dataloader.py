"""
Contains a dataloader for the cyclegan model.
"""

from typing import Optional
import numpy as np

from dataloaders import Dataset


class CycleGANDataloader:
    """Pools together MRI and CT scans for a cyclegan.

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
            mini_batch_sample_size: Optional[int] = 1,
            size_limit: Optional[int] = None
    ):
        self.loaders = [
            Dataset(data_root_path, dataset_name, "CT"),
            Dataset(data_root_path, dataset_name, "MR")
        ]
        if size_limit is None:
            self.loop_length = min(
                [len(x) for x in self.loaders]
            )
        else:
            self.loop_length = size_limit
        self.counter = 0
        self.batch_size = len(self.loaders[0][0])
        self.mini_batch_sample_size = mini_batch_sample_size

    def __len__(self):
        return self.loop_length

    def __iter__(self):
        return self

    def __next__(self):
        if self.counter == self.loop_length:
            self.counter = 0
        ct_batch = next(self.loaders[0])
        mr_batch = next(self.loaders[1])
        if self.mini_batch_sample_size is not None:
            minibatch_indexes = np.random.choice(
                range(self.batch_size),
                self.mini_batch_sample_size
            )
            return {
                "A": ct_batch[minibatch_indexes],
                "B": mr_batch[minibatch_indexes]
            }
        return {
            "A": ct_batch,
            "B": mr_batch
        }
