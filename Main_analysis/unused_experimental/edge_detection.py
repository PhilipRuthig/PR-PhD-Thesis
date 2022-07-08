import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage import feature
import tifffile as tf
'''
This script is meant to be used as a preprocessing step before registering an image. Registering an edge detedtion image is more
reliable than registering a raw image in my experience.
'''


#load input image
im=tf.imread(r"C:\Data\AJ003\results\step2_reference_image.tif").astype("float32")
# im=tf.imread(r"C:\Data\AJ003\AJ003_highres_downscaled_z_2.tif").astype("float32")
sigma=3

#####

# Compute sobel filter along all three axes
edges1 = ndi.sobel(im, axis=0)
edges2 = ndi.sobel(im, axis=1)
edges3 = ndi.sobel(im, axis=2)

# Average images and z score image
edges_sum = (edges1+edges2+edges3)/3
# edges_sum = edges_sum - np.mean(edges_sum)
# edges_sum = edges_sum / np.std(edges_sum)
edges_sum[edges_sum<0] = edges_sum[edges_sum<0]*(-1)

#invert image
# edges_sum = edges_sum*(-1)

#gauss img
edges_sum = ndi.gaussian_filter(edges_sum,sigma,)

#square to get rid of negative values
#edges_sum = np.sqrt(edges_sum*edges_sum)

####

# save raw image
tf.imsave(r"F:\temp\temp_raw.tif",im.astype("float32"))

# save resulting image
tf.imsave(r"F:\temp\temp.tif",edges_sum.astype("float32"))

print("Done!")