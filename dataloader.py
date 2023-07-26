import numpy as np
from glob import glob
import os

class Dataloader:
    def __init__(self, source, dataset, mode):
        source_dir = os.path.join(source, dataset)
        self.source_dir = source_dir
        self.mode = mode
        data_dir = glob(
            os.path.join(source_dir, mode, "*.npy")
        )[0]
        data_shape = tuple([int(x) for x in os.path.basename(data_dir).replace(".npy","").split("_")])
        self.batches = np.memmap(data_dir, dtype=np.float32, mode='r', shape=data_shape)
        self.counter = 0
        self.total_batches = data_shape[0]

    def __len__(self):
        return self.total_batches
    def __iter__(self):
        return self
    def __next__(self):
        if self.counter == self.total_batches:
            self.counter = 0
        batch = self.batches[self.counter]
        self.counter += 1
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