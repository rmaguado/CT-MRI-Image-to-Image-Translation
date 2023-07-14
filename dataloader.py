import numpy as np
from glob import glob
import os

DATASET_SOURCE = {
    "train" : "/nfs/home/clruben/workspace/nst/data/preprocessed/train/",
    "test" : "/nfs/home/clruben/workspace/nst/data/preprocessed/test/",
    "val" : "/nfs/home/clruben/workspace/nst/data/preprocessed/val/"
}

class DataMode:
    def __init__(self, source_dir, name="CT"):
        self.source_dir = source_dir
        self.name = name
        data_dir = glob(
            os.path.join(source_dir, name, "*.npy")
        )[0]
        data_shape = tuple([int(x) for x in os.path.basename(data_dir).replace(".npy","").split("_")])
        self.batches = np.memmap(data_dir, dtype=np.float32, mode='r', shape=data_shape)
        self.counter = 0
        self.total_batches = data_shape[0]

class Dataloader:
    def __init__(self, name):
        source_dir = DATASET_SOURCE[name]
        self.sources = [
            DataMode(source_dir, "MRI"), #mode 0
            DataMode(source_dir, "CT")   #mode 1
        ]
        self.mode = bool(1)
        self.total_batches = sum([x.total_batches for x in self.sources])

    def __len__(self):
        return self.total_batches
    def __iter__(self):
        return self
    def __next__(self):
        mode = self.mode
        current_source_number = self.sources[mode].counter
        if current_source_number == self.sources[mode].total_batches:
            for x in self.sources:
                x.counter = 0
            raise StopIteration
        
        batch = self.sources[mode].batches[current_source_number]
        self.sources[mode].counter += 1
        self.mode = not self.mode
        return batch, mode

if __name__ == "__main__":
    from tqdm import tqdm
    import time
    t0 = time.time()
    testloader = Dataloader("train")
    print(time.time()-t0)
    t0 = time.time()
    loop = tqdm(testloader)
    for batch, mode in loop:
        x = batch
    print(time.time()-t0)