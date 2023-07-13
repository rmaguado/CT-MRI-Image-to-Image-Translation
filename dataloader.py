import numpy as np
import gzip
from glob import glob
import time

DATASET_SOURCE = {
    "train" : "/nfs/home/clruben/workspace/nst/data/preprocessed/train/",
    "test" : "/nfs/home/clruben/workspace/nst/data/preprocessed/test/",
    "val" : "/nfs/home/clruben/workspace/nst/data/preprocessed/val/"
}

class Looper:
    def __init__(self, data):
        self.data = data
        self.counter = 0
    def __len__(self):
        return len(self.data)
    def next(self):
        next_item = self.data[self.counter]
        self.counter += 1
        if self.counter == len(self.data):
            self.counter = 0
        return next_item

class Dataloader:
    def __init__(self, name):
        source_dir = DATASET_SOURCE[name]
        mri = glob(source_dir + "/MRI/*.npy.gz")
        ct = glob(source_dir + "/CT/*.npy.gz")
        self.sources = {
            "MRI" : {
                "dirs" : mri, 
                "counter" : 0,
                "total" : len(mri)
            },
            "CT" : {
                "dirs" : ct, 
                "counter" : 0,
                "total" : len(ct)
            }
        }
        self.mode = "MRI"
        self.total_sources = len(mri) + len(ct)

    def __len__(self):
        return self.total_sources
    def __iter__(self):
        return self
    def __next__(self):
        mode = self.mode
        current_source_number = self.sources[mode]["counter"]
        if current_source_number == self.sources[mode]["total"]:
            for x in self.sources.keys():
                self.sources[x]["counter"] = 0
            raise StopIteration
        
        path = self.sources[mode]["dirs"][current_source_number]
        self.sources[mode]["counter"] += 1
        if self.mode == "MRI":
            self.mode = "CT"
        else:
            self.mode = "MRI"
        f = gzip.GzipFile(path, "r")
        batch = np.load(f)
        f.close()
        return batch, mode
