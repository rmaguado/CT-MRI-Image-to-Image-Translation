"""Dataloader for UNet.
"""

from typing import Optional
import numpy as np

from dataloaders import Dataset


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
            Dataset(data_root_path, dataset_name, "MR"),
            Dataset(data_root_path, dataset_name, "CT")
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
