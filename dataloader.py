import numpy as np
import gzip
from glob import glob
import time

DATASET_SOURCE = {
    "train" : "/nfs/home/clruben/workspace/nst/data/preprocessed/train/",
    "test" : "/nfs/home/clruben/workspace/nst/data/preprocessed/test/",
    "val" : "/nfs/home/clruben/workspace/nst/data/preprocessed/val/"
}

def load_gzip(path):
    f = gzip.GzipFile(path, "r")
    batch = np.load(f)
    f.close()
    return batch

class Dataloader:
    def __init__(self, name):
        source_dir = DATASET_SOURCE[name]
        mri = glob(source_dir + "/MRI/*.npy.gz")
        ct = glob(source_dir + "/CT/*.npy.gz")
        self.sources = {
            "MRI" : {
                "batches" : [load_gzip(p) for p in mri], 
                "counter" : 0,
                "total" : len(mri)
            },
            "CT" : {
                "batches" : [load_gzip(p) for p in mri], 
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
        
        batch = self.sources[mode]["batches"][current_source_number]
        self.sources[mode]["counter"] += 1
        if self.mode == "MRI":
            self.mode = "CT"
        else:
            self.mode = "MRI"
        return batch, mode
