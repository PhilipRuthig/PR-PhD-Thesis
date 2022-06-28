import numpy as np 
import h5py
import tifffile as tf
from matplotlib import pyplot as plt
from skimage.transform import resize
'''
Used to upscale e.g. the Allen Mouse Brain  atlas to the actual size of the full size dataset (after registration). 
'''


def primes(n):
    primfac = []
    d = 2
    while d*d <= n:
        while (n % d) == 0:
            primfac.append(d)
            n //= d
        d += 1
    if n > 1:
       primfac.append(n)
    return primfac

atlas_path =r"F:\Philip\AJ001\Registration\results\step3_atlas_image.tif" #aligned atlas image path
z_factor, y_factor, x_factor = 1,12,12 #factor of upscaling from low res to high res atlas
batch_shape   = (111,223,397)   #shape (Z,Y,X) of the batches being processed (in the original atlas)

atlas = tf.imread(atlas_path) #path to atlas image
atlas = atlas.astype("uint16")

try:
    f = h5py.File(r"F:\Philip\AJ001\Registration\results\atlas.hdf5","w")
except OSError:
    f.close()
    f = h5py.File(r"F:\Philip\AJ001\Registration\results\atlas.hdf5","w")

shape_upscaled = (atlas.shape[0]*z_factor,\
                  atlas.shape[1]*y_factor,\
                  atlas.shape[2]*x_factor)

f.create_dataset("atlas", shape=atlas.shape, dtype="uint16", chunks=True, maxshape=(None,None,None))
f.create_dataset("atlas_upscaled", shape=shape_upscaled, dtype="uint16", chunks=True, maxshape=(None,None,None))

f[u"atlas"][:] = atlas #insert atlas into h5

### initial coordinates and batch size
z_start = 0 
z_end   = batch_shape[0]
y_start = 0
y_end   = batch_shape[1]
x_start = 0
x_end   = batch_shape[2]

output_shape= z_end*z_factor-z_start*z_factor,\
              y_end*y_factor-y_start*y_factor,\
              x_end*x_factor-x_start*x_factor

while x_end <= atlas.shape[2]:
    
    print (z_start,y_start,x_start)

    #upscaling section of the image with nearest-neighbour interpolation (spline order=0)
    upscaled_section = resize(f[u"atlas"][z_start:z_end,\
                                          y_start:y_end,\
                                          x_start:x_end],
                                          output_shape=output_shape,
                                          anti_aliasing=False,
                                          order=0,
                                          mode="reflect",
                                          preserve_range=True,)

    f[u"atlas_upscaled"][z_start*z_factor:z_end*z_factor,\
                         y_start*y_factor:y_end*y_factor,\
                         x_start*x_factor:x_end*x_factor] = upscaled_section

    z_start = z_start + batch_shape[0]
    z_end   = z_start + batch_shape[0]
    
    # if z_end exceeds img shape: move y_start (and reset z_start)
    if z_end > atlas.shape[0]:
        y_start = y_start + batch_shape[1]
        y_end   = y_start + batch_shape[1]
        z_start = 0
        z_end   = batch_shape[0]
    
    # if z_end AND y_end exceed img shape: move x_start (and reset y_start and z_start)
    if y_end > atlas.shape[1]:
        x_start = x_start + batch_shape[2]
        x_end   = x_start + batch_shape[2]
        z_start = 0
        z_end   = batch_shape[0]
        y_start = 0
        y_end   = batch_shape[1]

print ("Done!")

# f.close()
# plt.imshow(f[u"atlas_upscaled"][200,:,:],cmap="prism",vmin=500,vmax=900,interpolation="none")
# plt.show()