import numpy as np
from glob import glob
import os

class Dataloader:
    def __init__(self, source, dataset, mode, data_augmentation=True):
        source_dir = os.path.join(source, dataset)
        self.source_dir = source_dir
        self.mode = mode
        self.data_augmentation = data_augmentation

        paths_to_database = glob(
            os.path.join(source_dir, mode, "*.npy")
        )
        if not paths_to_database:
            raise ValueError("Data not found. Check source directory provided.")

        data_dir = paths_to_database[0]
        data_shape = tuple(
            [
                int(x) for x in os.path.basename(
                    data_dir
                ).replace(".npy","").split("_")
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

    def __len__(self):
        return self.total_batches
    def __iter__(self):
        return self
    def __next__(self):
        if self.counter == self.total_batches:
            self.counter = 0
        batch = self.batches[self.counter]
        self.counter += 1
        if self.data_augmentation:
            return self.data_augmentation(batch)
        return batch

if __name__ == "__main__":
    from tqdm import tqdm
    import time
    t0 = time.time()
    source = "/nfs/home/clruben/workspace/nst/data/preprocessed/"
    testloader = Dataloader(source, "train", "CT")
    print(time.time()-t0)
    t0 = time.time()
    loop = tqdm(range(len(testloader)))
    for batch in loop:
        x = next(testloader)
    print(time.time()-t0)