from torch.utils.data import Dataset
import numpy as np
import gzip
import glob

DATASET_SOURCE = {
    "train" : "/nfs/home/clruben/workspace/nst/tempdata/preprocessed/train/",
    "test" : "/nfs/home/clruben/workspace/nst/tempdata/preprocessed/test/",
    "val" : "/nfs/home/clruben/workspace/nst/tempdata/preprocessed/val/"
}

class Dataset(Dataset):

    def __init__(self, name):
        source = DATASET_SOURCE[name]
        source_dirs = glob(source + "*.npy.gz")
        self.mri = []
        self.ct = []
        
        for path in source_dirs:
            f = gzip.GzipFile(path, "r")
            data = np.load(f)["arr_0"]
            f.close()
            samples += [x for x in data]
        data = np.array()

    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        return self.images[idx], 