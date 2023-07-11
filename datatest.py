from libtiff import TIFF
from glob import glob

img_dirs = glob("./tempdata/tiff_images/*.tif")
print(len(img_dirs))

tif = TIFF.open('./tempdata/tiff_images/ID_0000_AGE_0060_CONTRAST_1_CT.tif', mode='r')
image = tif.read_image()

print(image.shape)
