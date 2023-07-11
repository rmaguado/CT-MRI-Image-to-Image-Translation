from libtiff import TIFF
from glob import glob

img_dirs = glob("./tempdata/tiff_images/*.tif")
images = np.array(
    [
        TIFF.open(img_dir, mode='r').read_image() for img_dir in img_dirs
    ]
)
