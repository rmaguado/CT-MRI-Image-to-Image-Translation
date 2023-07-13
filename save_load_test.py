import numpy as np
import gzip
import time


def gzip_save(data):
    f = gzip.GzipFile("./random.npy.gz", "w")
    np.save(f, data)
    f.close()

def gzip_load():
    f = gzip.GzipFile("./random.npy.gz", "r")
    data = np.load(f)
    f.close()
    return data

def np_save(data):
    np.save("./random.npy", data)

def np_load():
    return np.load("./random.npy")

data = np.random.rand(10,10,1,512,512)
t0 = time.time()
np_save(data)
tf = time.time()
print("gzip save", tf-t0)

t0 = time.time()
data = np_load()
tf = time.time()
print("gzip load", tf-t0)

