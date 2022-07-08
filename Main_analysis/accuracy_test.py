import numpy as np
import tifffile as tf
'''
Used to add noise to an image for comparison. Thesis Figure 3.2.6.1.
'''

noise_meter = 1500

raw = tf.imread(r"D:\Light Sheet Data\manual accuracy computation\pr003-left\PR003_left_cropped_raw.tif")
#raw2 = tf.imread(r"D:\Light Sheet Data\manual accuracy computation\pr004-leftPR004_left_cropped_raw.tif")

# add noise
noise = np.random.normal(loc=0,scale=noise_meter,size=raw.shape)
img_noisy = raw.astype("uint16") + noise.astype("uint16")

tf.imsave(r"D:\Light Sheet Data\manual accuracy computation\pr003-left\PR003_left_cropped_raw_noisy.tif", img_noisy)
tf.imsave(r"D:\Light Sheet Data\manual accuracy computation\pr003-left\PR003_left_cropped_raw.tif", raw)