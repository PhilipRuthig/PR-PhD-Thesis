import h5py
import tifffile as tf
import numpy as np
'''
This script was intended to be used to crop a registered dataset to a certain allen mouse brain atlas region,
 e.g. the Auditory Cortex. This requires a dataset registered to the atlas.
'''

def crop_zeros(template,dat, bool=False): #crops zeros from outer rim of image. crops data just like template, must be same shape!
    import numpy as np
    if bool==True: 
        np.clip(template, 0, 1, out=template )
        np.clip(dat, 0, 1, out=dat)
        dat=dat.astype("bool")
        template=template.astype("bool")
    for axis in range(template.ndim):
        template = np.swapaxes(template, 0, axis)  # send i-th axis to front
        dat = np.swapaxes(dat, 0, axis)
        while np.all( template[0]==0 ):
            template = template[1:]
            dat = dat[1:]
        while np.all( template[-1]==0 ):
            template = template[:-1]
            dat = dat[:-1]
        template = np.swapaxes(template, 0, axis)  # send i-th axis to its original position
        dat = np.swapaxes(dat, 0, axis)
    return template,dat

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

def c():
    atlas.close()
    c1.close()
    print("closed h5 files")

regions = (735,251,816,847,954,) #list of ids of regions of interest 
batch_shape   = (112,30,31)   #shape (Z,Y,X) of the batches being processed (in the original atlas)

### initial coordinates and batch size
z_start = 0 
z_end   = batch_shape[0]
y_start = 0
y_end   = batch_shape[1]
x_start = 0
x_end   = batch_shape[2]

try:
    atlas= h5py.File(r"F:\Philip\AJ001\Registration\results\atlas.hdf5",mode="r+")
    c1   = h5py.File(r"F:\Philip\AJ001\AJ001_stitched\C01_converted",mode="r+")
except OSError:
    c()
    atlas= h5py.File(r"F:\Philip\AJ001\Registration\results\atlas.hdf5",mode="r+")
    c1   = h5py.File(r"F:\Philip\AJ001\AJ001_stitched\C01_converted",mode="r+")

atlas.create_dataset(u"bool_mask",shape=(atlas[u"atlas_upscaled"].shape),dtype="bool") #creates boolean mask

while x_end <= atlas[u"atlas_upscaled"].shape[2]: #create boolean mask for selected id's of regions

    #create subset of data
    current_atlas = atlas[u"atlas_upscaled"][z_start:z_end,\
                                             y_start:y_end,\
                                             x_start:x_end]

    current_bool = atlas[u"bool_mask"][z_start:z_end,\
                                       y_start:y_end,\
                                       x_start:x_end]
    
    for id in regions:
        current_bool[current_atlas==id]=True

    atlas[u"bool_mask"][z_start:z_end,\
                        y_start:y_end,\
                        x_start:x_end] = current_bool
    
    z_start = z_start + batch_shape[0]
    z_end   = z_start + batch_shape[0]
    
    # if z_end exceeds img shape: move y_start (and reset z_start)
    if z_end > atlas[u"atlas_upscaled"].shape[0]:
        y_start = y_start + batch_shape[1]
        y_end   = y_start + batch_shape[1]
        z_start = 0
        z_end   = batch_shape[0]
    
    # if z_end AND y_end exceed img shape: move x_start (and reset y_start and z_start)
    if y_end > atlas[u"atlas_upscaled"].shape[1]:
        x_start = x_start + batch_shape[2]
        x_end   = x_start + batch_shape[2]
        z_start = 0
        z_end   = batch_shape[0]
        y_start = 0
        y_end   = batch_shape[1]

#apply boolean array to image

# c1[u"t00000/s00/0/cells"]
# atlas[u"atlas_upscaled"]
# atlas[u"bool_mask"]
