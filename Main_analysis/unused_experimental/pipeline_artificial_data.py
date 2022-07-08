# In[Initiation of variables, packages and functions]
from __future__ import division
import numpy as np # for array stuff
import matplotlib.pyplot as plt # for plotting
import scipy.ndimage as ndi # n-dimensional scientific image processing tools
import scipy.signal as sig # additional data processing stuff
import tifffile as tf # for opening & saving tifs
import skimage.morphology as morph # for creating spheres as structuring elements
from skimage.transform import downscale_local_mean # downsampling of the image for faster analysis
import time # timekeeping
from skimage.feature import peak_local_max # for finding local maxima
import random
import math as m
'''
Performs gabor annulus annotation on artificial data for proof-of-function
'''
def gaborkernel(edge_length, sigma, freq, phase, radius, z_scale_factor):
    pi=m.pi

    if edge_length%2==0:
        edge_length = edge_length+1

    size_z=np.arange(0,edge_length,1)
    size_y=np.arange(0,edge_length,1)
    size_x=np.arange(0,edge_length,1)
    
    z,y,x = np.meshgrid(size_z, size_y, size_x)
    z,y,x = z-(len(size_z)//2),y-(len(size_y)//2),x-(len(size_x)//2)
    y = y*z_scale_factor
    
    A = (2*pi*sigma**2)
    r = np.sqrt(np.power(z,2) + np. power(y,2) + np.power(x,2))
    
    kernel_real = (1/A)*np.exp(-1*pi*((np.power((r-radius),2))/sigma**2)) * (np.cos((freq*(r-radius)*2*pi+phase)))
    kernel_imag = (1/A)*np.exp(-1*pi*((np.power((r-radius),2))/sigma**2)) * (np.sin((freq*(r-radius)*2*pi+phase)))
    
    #inverting kernels
    kernel_real = kernel_real*(-1)
    kernel_imag = kernel_imag*(-1)
        
    return kernel_real, kernel_imag

start_time = time.time()
### options on what the script should do
colormap  = "gray"
plot      = True #Toggle plotting of preliminary 2d slices during segmentation
size      = 8 #size of figures to be displayed
save      = False #Toggle saving of resulted images
save_path = r"C:\Users\pr55gone\Documents\Cloud\11. Semester\Master_Thesis\20180329_Zeiss_Data\artificial_data"

### variables that will need tweaking depending on the image being processed
scaling       = 2 #factor of how much each axis of the original image will be scaled down
gauss_sigma   = 0.5 #sigma for gaussian blur for foreground image

### options for generated sample image
image_edge_length = 400 #xyz of image
n_cells = 50 #number of cells
background_intensity = 10000 #background intensity
cell_intensity = 10300
cell_radius_range = range(27,28,1) #list of possible cell radiuses in px. Factor in scaling!
noise_meter = 70 # standard deviation of gaussian function where noise is randomly picked from. bigger = more noise

### gabor filter variables. need to be adapted to cell size, data & quality
gabor_edge_length = 70   #kernel size(x,y,z)
gabor_freq        = 1/10 #frequency of the wave component of GA
gabor_sigma       = 10   #gaussian deviation
gabor_min_radius  = 12   #first gabor radius to be calculated
gabor_max_radius  = 13   #last gabor radius to be calculated
gabor_phase       = 3    #quarter wavelength (inverse of gabor_freq) offset to enhance contrast of filter
z_factor          = 1    #factor of how much the z axis is compressed in microscopy data. 1 for isotropic data
r_maxima          = 8    #radius of footprint for detecting local maxima in the gabor filtered image

### specific to artificial data
gabor_threshold = 81000000 # high pass filter threshold for cell centres.

footprint_maxima = morph.ball(r_maxima)   # structuring element for detecting local maxima in gabor annulus filtered image

# In[Loading and preprocessing of image]
img = np.zeros((image_edge_length,image_edge_length,image_edge_length), dtype="uint16")
img = np.pad(array=img,
             pad_width=cell_radius_range[-1]+2,
             mode="constant",
             constant_values=0)

img[img<1] = background_intensity

for n_cells in range(n_cells):
    
    r_cell = cell_radius_range[random.randint(0,len(cell_radius_range)-1)]
    cell_body = morph.ball(r_cell)
    
    randx = random.randint(r_cell,image_edge_length+r_cell)
    randy = random.randint(r_cell,image_edge_length+r_cell)
    randz = random.randint(r_cell,image_edge_length+r_cell)
    
    cell_body_array = np.zeros_like(img)
    cell_body_array[randx-r_cell:randx+r_cell+1,
                    randy-r_cell:randy+r_cell+1,
                    randz-r_cell:randz+r_cell+1] = cell_body
    
    #adding single cell to whole image
    img = img + cell_body_array
    print ("inserted cell #" + str(n_cells))

# cropping image down to original size
img = img[cell_radius_range[-1]:-cell_radius_range[-1],
          cell_radius_range[-1]:-cell_radius_range[-1],
          cell_radius_range[-1]:-cell_radius_range[-1]]

# set intersections of cells to respective intensity
img[img>background_intensity] = cell_intensity

# add noise
noise = np.random.normal(loc=0,scale=noise_meter,size=img.shape)
img_noisy = img + noise

#downsample
img = downscale_local_mean(img_noisy, (scaling,scaling,scaling))
img = img.astype("uint16")

if plot == True:
    print ("printing raw image")
    plt.figure(figsize=(size,size))
    plt.imshow(img[0],interpolation='none', cmap=colormap)
    plt.colorbar()
    plt.show()

print (str(int(time.time() - start_time)) + " seconds for initialization.")
# In[Gaussian Blur]
# applying a moderate gaussian filter to create a foreground image
print ("applying gaussian filter")
gaussedhucd = ndi.filters.gaussian_filter(img, sigma=gauss_sigma)

if plot == True:
    print ("printing gauss filtered images")
    plt.figure(figsize=(size,size))
    plt.imshow(gaussedhucd[0],interpolation='none', cmap=colormap)
    plt.colorbar()
    plt.show()

print (str(int(time.time() - start_time)) + " seconds for applying gaussian blur.")

# In[Filtering image with different Gabor Annulus radii]
print ("performing gabor annulus filtering")

# padding image
pad_size = gabor_max_radius*2
padded_hucd = np.pad(img,pad_size,mode='constant',constant_values=background_intensity)

gaborimages = {}
gaborfilters = {}

for radius in range(gabor_min_radius,gabor_max_radius):

    # adjusting gabor radius
    gabor_radius = radius
    
    # creating filter kernels
    gabor_kernel_real, gabor_kernel_imag = gaborkernel(
                                                         edge_length=gabor_edge_length,
                                                         sigma=gabor_sigma,
                                                         freq=gabor_freq,
                                                         radius=gabor_radius,
                                                         z_scale_factor=z_factor,
                                                         phase=gabor_phase
                                                       )
    
    if plot==True:
        print ("real part of gabor filter at x/2:")
        plt.figure(figsize=(size,size))
        plt.pcolormesh(gabor_kernel_real[gabor_edge_length//2,:,:], cmap=colormap)
        plt.colorbar()
        plt.show()
        
        print ("real part of gabor filter at y/2:")
        plt.figure(figsize=(size,size))
        plt.pcolormesh(gabor_kernel_real[:,gabor_edge_length//2,:], cmap=colormap)
        plt.colorbar()
        plt.show()
    
    # applying gabor filter on valid part of the image, also cropping it back to original size
    gaborimg_real = sig.fftconvolve(padded_hucd, gabor_kernel_real, mode="same")
    
    # cropping image back to original shape
    gaborimg_real = gaborimg_real[pad_size:-pad_size,
                                  pad_size:-pad_size,
                                  pad_size:-pad_size]
    
    # adjusting img intensity. image as dtype float because it will give maxima larger than 1px³ otherwise
    gaborimg_real = gaborimg_real*gaussedhucd
    gaborimg_real = gaborimg_real.astype("float32")
    
    if plot==True:
        print ("printing gabor image for radius " + str(radius))
        plt.figure(figsize=(size,size))
        plt.pcolormesh(gaborimg_real[0,:,:], cmap=colormap)
        plt.colorbar()
        plt.show()
        
    gaborimages["radius_" + str(radius)]  = gaborimg_real
    gaborfilters["radius_" + str(radius)] = gabor_kernel_real

print (str(int(time.time() - start_time)) + " seconds for gabor annulus filtering.")
# In[Processing filtered images]

gauss_cellcenters = {}
cellcenters = {}

for i in range(0,len(gaborimages)):
    # marking local maxima
    centers = peak_local_max(
                            image=gaborimages["radius_" + str(gabor_min_radius+i)],
                            min_distance=7,
                            indices=False,
                            footprint=footprint_maxima,
                            exclude_border=0
                            )
    
    # remove marked centers at image borders
    centers[0,:,:] = 0
    centers[:,0,:] = 0
    centers[:,:,0] = 0
    centers[-1,:,:] = 0
    centers[:,-1,:] = 0
    centers[:,:,-1] = 0

    # label cells and print number of cells
    cellcenters_labeled, n_cells_gabor = ndi.label(centers)
    
    # threshold identified cell centres with corresponding gabor image - delete cell centres  that don't reach threshold
    for x in np.nditer(cellcenters_labeled[cellcenters_labeled!=0]):
        if x == 0:
            break
        if np.any(gaborimages["radius_" + str(gabor_min_radius+i)][cellcenters_labeled==x] < gabor_threshold)==True:
            cellcenters_labeled[cellcenters_labeled==x] = 0

    # relabel filtered cellcenters
    centers = np.zeros_like(cellcenters_labeled,dtype="uint16")
    centers[cellcenters_labeled>0] = cellcenters_labeled[cellcenters_labeled>0]
    centers[centers>0] = 1
    cellcenters_labeled, n_cells_gabor = ndi.label(centers)
    
    print ("number of cell centers identified by gabor annulus with radius " + str(gabor_min_radius+i) + ":" + str(n_cells_gabor))
    
    # insert cell centers at 1.5*max intensity into gaussed image
    gaussedhucd_cellcenters = np.copy(gaussedhucd)
    gaussedhucd_cellcenters[centers==True] = int(np.max(gaussedhucd)+np.max(gaussedhucd)/2)
    
    # insert cell centers at 1.5*max intensity into gabor filtered image
    gaborimg_real_cellcenters = np.copy(gaborimg_real)
    gaborimg_real_cellcenters[centers==True] = int(np.max(gaborimg_real)+np.max(gaborimg_real)/2) 
    
    gauss_cellcenters["radius_" + str(i+gabor_min_radius)] = gaussedhucd_cellcenters
    cellcenters["radius_" + str(i+gabor_min_radius)] = centers

print (str(int(time.time() - start_time)) + " seconds for finding local maxima in gabor images.")

# In[Saving of the resulting images and used variables]

if save == True:
    with open(r"C:\Users\pr55gone\Documents\Cloud\11. Semester\Master_Thesis\20180329_Zeiss_Data\artificial_data\variables.txt", "w") as variables:
        variables.write("scaling factor = " + str(scaling) + "\n")
        variables.write("gaussian sigma = " + str(gauss_sigma) + "\n")
        variables.write("gabor edge length = " + str(gabor_edge_length) + "\n")
        variables.write("gabor frequency = " + str(gabor_freq) + "\n")
        variables.write("gabor phase = " + str(gabor_phase) + "\n")
        variables.write("gabor sigma = " + str(gabor_sigma) + "\n")
        variables.write("gabor radius = " + str(gabor_radius) + "\n")
        variables.write("maxima detection radius = " + str(r_maxima) + "\n")
        variables.write("gabor threshold = " + str(gabor_threshold) + "\n")
    tf.imsave(file= save_path + r"\kernel_real.tif", data=gabor_kernel_real.astype("float32"))
    tf.imsave(file= save_path + r"\kernel_imag.tif", data=gabor_kernel_imag.astype("float32"))
    tf.imsave(file= save_path + r"\hucd_gaussfiltered.tif", data=gaussedhucd.astype("uint16"))
    tf.imsave(file= save_path + r"\rawimage.tif", data=img_noisy.astype("uint16"))

    for i in range(0,len(gaborimages)):
        tf.imsave(file = save_path + r"\r" + str(i+gabor_min_radius) + r"_kernel_real.tif",
                  data = gaborfilters["radius_" + str(i+gabor_min_radius)].astype("float32"))
        tf.imsave(file = save_path + r"\r" + str(i+gabor_min_radius) + r"_gaborimage.tif",
                  data = gaborimages["radius_" + str(i+gabor_min_radius)].astype("float32"))
        tf.imsave(file = save_path + r"\r" + str(i+gabor_min_radius) + r"_gaussed_cellcenters.tif",
                  data = gauss_cellcenters["radius_" + str(i+gabor_min_radius)].astype("uint16"))
        tf.imsave(file = save_path + r"\r" + str(i+gabor_min_radius) + r"_cellcenters.tif",
                  data = cellcenters["radius_" + str(i+gabor_min_radius)].astype("uint16"))

print (str(int(time.time() - start_time)) + " seconds for labelling, and saving.")

# In[stuff]