import numpy as np
from glob import glob
import os

DATASET_SOURCE = {
    "train" : "/nfs/home/clruben/workspace/nst/data/preprocessed/train/",
    "test" : "/nfs/home/clruben/workspace/nst/data/preprocessed/test/",
    "val" : "/nfs/home/clruben/workspace/nst/data/preprocessed/val/"
}

class Dataloader:
    def __init__(self, name, batch_size, img_size):
        source_dir = DATASET_SOURCE[name]
        mri_dir = glob(source_dir + "/MRI/*.npy")[0]
        ct_dir = glob(source_dir + "/CT/*.npy")[0]
        mri_data_shape = os.path.basename(mri_dir).replace(".npy","").split("_")
        ct_data_shape = os.path.basename(ct_dir).replace(".npy","").split("_")
        self.sources = {
            "MRI" : {
                "batches" : np.memmap(mri_dir, dtype=np.float32, mode='r', size=mri_data_shape), 
                "counter" : 0,
                "total_batches" : mri_data_shape[0]
            },
            "CT" : {
                "batches" : np.memmap(ct_dir, dtype=np.float32, mode='r', size=ct_data_shape),
                "counter" : 0,
                "total_batches" : ct_data_shape[0]
            }
        }
        self.mode = "MRI"
        self.total_sources = mri_data_shape[0] + ct_data_shape[0]

    def __len__(self):
        return self.total_sources
    def __iter__(self):
        return self
    def __next__(self):
        mode = self.mode
        current_source_number = self.sources[mode]["counter"]
        if current_source_number == self.sources[mode]["total_batches"]:
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

if __name__ == "__main__":
    from tqdm import tqdm
    testloader = Dataloader("test")
    loop = tqdm(testloader)
    for batch, mode in loop:
        x = batch