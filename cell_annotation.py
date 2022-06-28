# -*- coding: utf-8 -*-
"""
@author: Philip Ruthig, PhD Student at Leipzig University & MPI for Cognitive and Brain Sciences.
This script takes h5 or 3d tiff files as input and annotates round structures in them (e.g. cell centres, cell nuclei).
Afterward, a spatial cell center distribution is calculated and the results as well as the processed files are saved in a single h5
file for every input file. The pipeline is designed to be runnable on an average office workstation (recommended: 16GB RAM) 
through batch processing.
For replication of my PhD Thesis results, couple this script with cell_centre_analysis.py.
"""

### Initiation of variables, packages and functions

import os
import time
import datetime
import h5py
import numpy as np
import scipy.ndimage as ndi
import scipy.signal as sig
import math as m
import matplotlib.pyplot as plt
from skimage.transform import downscale_local_mean
from skimage.feature import peak_local_max
import skimage.morphology as morph
import tifffile as tf

def closeall():
    results.close()
    # img_file.close()
    return None

def gaborkernel(edge_length,sigma, freq, phase, radius, z_scale_factor):
    '''
    Returns two gabor kernels (real and imaginary) for spheroid detection. Can be convolved with an image (using scipy.signal.fftconvolve)
    
    edge_length: edge length of the kernel array
    sigma: sigma of gaussian part of the gabor kernel
    freq: frequency of the complex plane wave
    phase: phase displacement in pixels
    radius: radius of how much the gaussian wave is displaced from the origin
    z_scale_factor: factor of how much z axis is compressed. 1 for isotropic data
    '''
    if edge_length%2==0:
        edge_length = edge_length+1
    
    size_z=np.arange(0,edge_length,1)
    size_y=np.arange(0,edge_length,1)
    size_x=np.arange(0,edge_length,1)
    
    z,y,x = np.meshgrid(size_z, size_y, size_x)
    z,y,x = z-(len(size_z)//2),y-(len(size_y)//2),x-(len(size_x)//2)
    y = y*z_scale_factor
    
    A = (2*m.pi*sigma**2)
    r = np.sqrt(np.power(z,2) + np. power(y,2) + np.power(x,2))
    
    kernel_real = (1/A)*np.exp(-1*m.pi*((np.power((r-radius),2))/sigma**2)) * (np.cos((freq*(r-radius)*2*m.pi+phase)))
    kernel_imag = (1/A)*np.exp(-1*m.pi*((np.power((r-radius),2))/sigma**2)) * (np.sin((freq*(r-radius)*2*m.pi+phase)))
            
    #inverting kernels
    kernel_real = kernel_real*(-1)
    kernel_imag = kernel_imag*(-1)
        
    return kernel_real, kernel_imag

def cell_centre_distribution(bool_input,reach,sparsity_factor=1):
    '''
    Computes a mean distribution of cells in a given boolean array in a given edge length cube. Cycles through all TRUE pixels
    and checks the surroundings in a cube with the edge length "reach".

    bool_input: boolean numpy array input
    reach: edge length of the cube
    sparsity_factor: subsampling factor. 1 means every cell, 10 every 10th cell
    '''

    struct = np.zeros((3,3,3))
    struct[1,1,1] = 1
    
    centres_labeled, n_cells_labeled = ndi.label(bool_input, structure=struct)

    invalid_area_mask = np.ones_like(bool_input,dtype="bool") #define valid part of data - exclude outer rim
    invalid_area_mask[0:reach//2,:,:] = False
    invalid_area_mask[-reach//2:,:,:] = False
    invalid_area_mask[:,-reach//2:,:] = False
    invalid_area_mask[:,0:reach//2,:] = False
    invalid_area_mask[:,:,-reach//2:] = False
    invalid_area_mask[:,:,0:reach//2] = False
    
    for i in range(0,n_cells_labeled,sparsity_factor): #iterate over all cells
        if i == 0: #initialize analysis
            resultarray = np.zeros((reach,reach,reach)) #initialize results array
            n_valid_cells = 0
            n_invalid_cells = 0
            continue
         
        if i%500==0: #timekeeping
            print ("running analysis on cell number " + str(i) + " of " + str(n_cells_labeled))

        clocz,clocy,clocx = np.nonzero(centres_labeled==i) #get active cell coordinates
            
        if invalid_area_mask[clocz[0],clocy[0],clocx[0]]==True:
            n_valid_cells += 1
            tmpdata = bool_input[clocz[0]-reach//2:clocz[0]+reach//2,clocy[0]-reach//2:clocy[0]+reach//2,clocx[0]-reach//2:clocx[0]+reach//2]
        else:
            n_invalid_cells += 1
            continue
        
        resultarray = resultarray + tmpdata #add tmp data to complete results array
        
    resultarray[reach//2,reach//2,reach//2] = 0 #delete reference cell
    return resultarray

### general script options
plot          = False          #Toggle plotting of preliminary 2d slices
colormap      = "gray"         #colormap for 2d plots
size          = 8              #size of displayed plots
save          = True           #Toggle saving of results
compression   = 2              #lossless compression (gzip) of results h5 file, between 1-9. Higher = more compression
cell_dist     = True           #toggle if cell centre distribution should be computed or not (takes a while)
shutdown      = False          #Toggle shutdown after script is finished
#open_path     = r"E:\new_substacks_gesine"  #path to folder with images (in h5 format if h5 == True)

### tuple of image names to be processed
img_paths = (
# r"/home/ruthig/nas/Gesine_Muellerg/new_substacks_gesine/PR009_l_ACx/PR009_l_ACx.h5",
# r"/home/ruthig/nas/Gesine_Muellerg/new_substacks_gesine/PR009_r_ACx/PR009_r_ACx.h5",
# r"/home/ruthig/nas/Gesine_Muellerg/new_substacks_gesine/PR010_l_ACx/PR010_l_ACx.h5",
# r"/home/ruthig/nas/Gesine_Muellerg/new_substacks_gesine/PR010_r_ACx/PR010_r_ACx.h5",
# r"/home/ruthig/nas/Gesine_Muellerg/new_substacks_gesine/PR012_l_ACx/PR012_l_ACx.h5",
# r"/home/ruthig/nas/Gesine_Muellerg/new_substacks_gesine/PR012_r_ACx/PR012_r_ACx.h5",
# r"/home/ruthig/nas/Gesine_Muellerg/new_substacks_gesine/PR014_l_ACx/PR014_l_ACx.h5",
# r"/home/ruthig/nas/Gesine_Muellerg/new_substacks_gesine/PR014_r_ACx/PR014_r_ACx.h5",
# r"/home/ruthig/nas/Gesine_Muellerg/new_substacks_gesine/PR016_l_ACx/PR016_l_ACx.h5",
# r"/home/ruthig/nas/Gesine_Muellerg/new_substacks_gesine/PR016_r_ACx/PR016_r_ACx.h5",
# r"/home/ruthig/nas/Gesine_Muellerg/new_substacks_gesine/PR017_l_ACx/PR017_l_ACx.h5",
# r"/home/ruthig/nas/Gesine_Muellerg/new_substacks_gesine/PR017_r_ACx/PR017_r_ACx.h5",
# r"/home/ruthig/nas/Gesine_Muellerg/new_substacks_gesine/PR018_l_ACx/PR018_l_ACx.h5",
# r"/home/ruthig/nas/Gesine_Muellerg/new_substacks_gesine/PR018_r_ACx/PR018_r_ACx.h5",
r"/data/ssd2/Philip/cs002_l_AC.tif",
r"/data/ssd2/Philip/cs002_r_AC.tif",
r"/data/ssd2/Philip/cs003_l_AC.tif",
r"/data/ssd2/Philip/cs003_r_AC.tif",
r"/data/ssd2/Philip/cs004_l_AC.tif",
r"/data/ssd2/Philip/cs004_r_AC.tif",)


img_crops = []
img_crops.append((20,106,40,1050,600,1800))#pr009_l
img_crops.append((110,190,190,1350,110,1500))#pr009_r
img_crops.append((210,325,370,1800,430,1900))#pr012_l
img_crops.append((180,280,800,2000,400,1950))#pr012_r
img_crops.append((217,310,550,1800,250,2010))#pr014_l
img_crops.append((180,280,500,1800,130,2050))#pr014_r
img_crops.append((230,315,1000,2250,100,1900))#pr017_l
img_crops.append((195,290,600,1900,100,1800))#pr017_r
img_crops.append((80,190,400,1600,80,1400))#pr010_l
img_crops.append((190,290,750,2100,20,1500))#pr010_r
img_crops.append((70,180,550,1900,400,2000))#pr018_l
img_crops.append((80,185,550,1900,400,2000))#pr018_l

save_path     = r"/data/ssd2/Philip/fifthrun_AC_only/"  #path to results folder
checkprogress = 3               #check progress of script every x batches
batch_shape   = (160,430,430)   #shape (Z,Y,X) of the batches being processed. Larger is faster but also requires more RAM. Use even numbers (160,430,430)
h5            = False           #is the input image h5? no = 3D tif
h5path        = u't00000/s01/0/cells' #s00 = AF, s01 = HuCD, s02 = TOPRO, s03 = MBP

### variables that will need tweaking depending on the image being processed
gauss_sigma   = 0.7   #sigma for gaussian blurred image

hucd=True #True means HuCD data. False means Topro data.

### gabor filter variables. need to be adapted to cell size, data & quality
if hucd==True:
    edge         = 40   #gabor kernel edge length (x=y=z)
    gabor_freq   = 1/10 #frequency of wave component
    gabor_phase  = 3.7  #wave component offset
    gabor_sigma  = 8   #gaussian deviation #7
    gabor_radius = 9    #donut radius       #10  
    z_factor     = 7  #factor of how much the z axis is compressed in microscopy data. 1 for isotropic data
    tissue_thresh= 1000  #intensity threshold to check cells are in tissue
    r_maxima     = 6    #radius of footprint for detecting local maxima in the gabor filtered image #6
    min_distance = 4    #min distance between local maxima #4

if hucd==False:
    edge         = 40   #gabor kernel edge length (x=y=z)
    gabor_freq   = 1/10 #frequency of wave component
    gabor_phase  = 3.7  #wave component offset
    gabor_sigma  = 8   #gaussian deviation #7
    gabor_radius = 4.5    #donut radius       #10  
    z_factor     = 7  #factor of how much the z axis is compressed in microscopy data. 1 for isotropic data
    tissue_thresh= 700  #intensity threshold between background max intensity and tissue min intensity. To check if detected cells are in tissue
    r_maxima     = 6    #radius of footprint for detecting local maxima in the gabor filtered image #6
    min_distance = 4    #min distance between local maxima #4

### data analysis variables
reach = 100 #number of surrounding pixels taken into account for cell distribution analysis
sparsity_factor = 5 #subsampling for cell distribution analysis

### initializing variables and structuring elements
footprint_maxima = morph.ball(r_maxima)   # structuring element for detecting local maxima in gabor annulus filtered image

n_image=-1
for img_path in img_paths: # execute for all images in list. Must be inside directory of open_path
    n_image +=1
    n_iter = 0                                # number of batches processed
    n_cells = 0                               # total number of identified cells
    print ("starting analysis for " + img_path)
     
    ### initialize variables for timekeeping and timestamping data
    start_time = time.time()
    timestamp = datetime.datetime.fromtimestamp(start_time).strftime(r'%Y%m%d_%H.%M')

    # open raw image file and dataset
    if h5 == True:
        img_file = h5py.File(img_path, "r")
        img_shape = img_file[h5path].shape 

    if h5 == False:    # open tif
        img_file=tf.imread(img_path)
        img_shape = img_file.shape


    #approximate batches to be computed
    n_approx_batches = (img_shape[0]*img_shape[1]*img_shape[2])/((batch_shape[0]-edge)*(batch_shape[1]-edge)*(batch_shape[2]-edge))
    print ("approximate number of batches to be calculated: " + str(int(n_approx_batches)))
    
    if save == True:
        # initialize empty h5 file for results
        results = h5py.File(save_path + r"/results_" + str(img_path[-14:-3]) + "_" + str(timestamp) + r".h5", "w")
        results.create_dataset(u"gaussed_image",
                               dtype="uint16",
                               chunks=True,
                               compression=compression, 
                               shape=(img_shape))
    
    print (str(int(time.time() - start_time)) + " seconds for loading h5 file.")
    
    # In[Main Sequence]
    # main sequence starts here. Iterates through z, then through y (and z) and then through x (and y and z) axis batch-wise.
    # The process is finished when the x-coordinates of the batch exceed the x edge length of the original Image.
    
    ### temp:
    if h5 == False:
        batch_shape = img_shape
    
    ### initial coordinates and batch size
    z_start = 0 
    z_end   = batch_shape[0]
    y_start = 0
    y_end   = batch_shape[1]
    x_start = 0
    x_end   = batch_shape[2]
    
    # ### initial coordinates and batch size
    # z_start = img_crops[n_image][0]-20
    # z_end   = img_crops[n_image][1]+20
    # y_start = img_crops[n_image][2]-20
    # y_end   = img_crops[n_image][3]+20
    # x_start = img_crops[n_image][4]-20
    # x_end   = img_crops[n_image][5]+20
    
    while x_end <= img_shape[2]:
    # while n_iter<1: #uncomment this line if you only want to do a specific amount of batches for testing purposes
        # timekeeping 
        if n_iter>0 and n_iter%checkprogress == 0: #print mid-analysis results every x iterations
            print ("number of total cells found so far: " + str(n_cells))
            print ("Elapsed time: " + str(int((time.time() - start_time)/60)) + " minutes")
            approx_rest_time = int(((time.time() - start_time)/n_iter)*(n_approx_batches-n_iter))
            print ("Approximate time until analysis is finished: " + str(approx_rest_time/60) + " minutes")
            
        n_iter +=1
        print ("start processing of batch #" + str(n_iter))

        if h5==True:
            current_img = img_file[h5path][z_start:z_end,y_start:y_end,x_start:x_end]
        if h5==False:
            current_img = img_file[z_start:z_end,y_start:y_end,x_start:x_end]
        
        # Preprocessing of Image
        img = current_img #renaming to img
        img = img.astype("uint16")
        img = img.astype("float32")

        # Gaussian Blur
        # applying a moderate gaussian filter
        gaussed_img = ndi.filters.gaussian_filter(img, sigma=gauss_sigma)
        # gaussed_img = gaussed_img.astype("float32")
        
        if plot == True:
            print ("printing gauss filtered image")
            plt.figure(figsize=(size,size))
            plt.imshow(gaussed_img[gaussed_img.shape[0]//2,:,:],interpolation='none', cmap=colormap)
            plt.colorbar()
            plt.show()
        
        # creating filter kernel
        gabor_kernel_real, gabor_kernel_imag = gaborkernel(
                                                           edge_length=edge,
                                                           sigma=gabor_sigma,
                                                           freq=gabor_freq,
                                                           radius=gabor_radius,
                                                           z_scale_factor=z_factor,
                                                           phase=gabor_phase
                                                          )
        
        # applying gabor filter on valid part of the image (returning an image of the same shape)
        # gabor_edge_length//2 on every end of every axis has to be cropped when saving
        gaborimg_real = sig.fftconvolve(img, gabor_kernel_real, mode="same")

        if hucd==False:
            gaborimg_real=gaborimg_real*(-1) 
        
        gaborimg_real = gaborimg_real.astype("float32")
        
        if plot==True:
            print ("printing uncropped gabor image")
            plt.figure(figsize=(size,size))
            plt.imshow(gaborimg_real[edge//2+1,:,:], cmap=colormap)
            plt.colorbar()
            plt.show() 

        # marking local maxima
        centers = peak_local_max(
                                image=gaborimg_real,
                                min_distance=min_distance,
                                indices=False,
                                footprint=footprint_maxima,
                                exclude_border=0,
                                )
        
        # threshold centers according to tissue background intensity
        centers[gaussed_img<tissue_thresh]=0

        # crop center coordinates image to valid part
        centers = centers[edge//2:-edge//2,edge//2:-edge//2,edge//2:-edge//2]
        
        # label and count cell centers
        cellcenters_labeled, n_cells_batch = ndi.label(centers)

        # insert cell centers at max intensity into gaussed image
        gaussed_img_cellcenters = np.copy(gaussed_img[edge//2:-edge//2,edge//2:-edge//2,edge//2:-edge//2])
        gaussed_img_cellcenters[centers==True] = 65535
        
        # insert cell centers at max intensity into gabor filtered image
        gaborimg_cellcenters = np.copy(gaborimg_real[edge//2:-edge//2,edge//2:-edge//2,edge//2:-edge//2])
        gaborimg_cellcenters[centers==True] = 65535

        #crop gaussed image and gabor image to valid analyzed part
        gaborimg_real = gaborimg_real[edge//2:-edge//2,edge//2:-edge//2,edge//2:-edge//2]
        gaussed_img = gaussed_img[edge//2:-edge//2,edge//2:-edge//2,edge//2:-edge//2]

        # create datasets for cellcenters, gabor image and gaussimg/gaborimg + cellcenters
        if n_iter == 1 and save == True:
            results.create_dataset(u"cellcenters",
                           dtype="uint8",
                           chunks=True,
                           compression=compression, 
                           shape=(img_shape))

            results.create_dataset(u"gabor_cellcenters",
                           dtype="float32",
                           chunks=True,
                           compression=compression, 
                           shape=(img_shape))
    
            results.create_dataset(u"gaussed_cellcenters",
                           dtype="float32",
                           chunks=True,
                           compression=compression, 
                           shape=(img_shape))

            results.create_dataset(u"gabor_image",
                           dtype="float32",
                           chunks=True,
                           compression=compression, 
                           shape=(img_shape))
            
            results.create_dataset(u"result_array",
                           dtype="uint16",
                           chunks=True,
                           compression=compression, 
                           shape=(reach,reach,reach))
                 
            results[u"gabor_filter"] = gabor_kernel_real
    
        if save == True:# insert batches into h5 datasets
            results[u"gaussed_image"][(z_start)+edge//2:(z_end)-edge//2,(y_start)+edge//2:(y_end)-edge//2,(x_start)+edge//2:(x_end)-edge//2] = gaussed_img
            results[u"gabor_image"][(z_start)+edge//2:(z_end)-edge//2,(y_start)+edge//2:(y_end)-edge//2,(x_start)+edge//2:(x_end)-edge//2] = gaborimg_real
            results[u"cellcenters"][(z_start)+edge//2:(z_end)-edge//2,(y_start)+edge//2:(y_end)-edge//2,(x_start)+edge//2:(x_end)-edge//2] = centers
            results[u"gaussed_cellcenters"][(z_start)+edge//2:(z_end)-edge//2,(y_start)+edge//2:(y_end)-edge//2,(x_start)+edge//2:(x_end)-edge//2] = gaussed_img_cellcenters
            results[u"gabor_cellcenters"][(z_start)+edge//2:(z_end)-edge//2,(y_start)+edge//2:(y_end)-edge//2,(x_start)+edge//2:(x_end)-edge//2] = gaborimg_cellcenters
        
        n_cells = n_cells + n_cells_batch
        print("cells found this batch: " + str(n_cells_batch))
        z_start = z_start + np.shape(current_img)[0]-2*edge
        z_end   = z_start + batch_shape[0]
        
        # if z_end exceeds img shape: move y_start (and reset z_start)
        if z_end >= img_shape[0]:
            y_start = y_start + np.shape(current_img)[1]-2*edge
            y_end   = y_start + batch_shape[1]
            z_start = 0
            z_end   = batch_shape[0]
        
        # if z_end AND y_end exceed img shape: move x_start (and reset y_start and z_start)
        if y_end >= img_shape[1]:
            x_start = x_start + np.shape(current_img)[2]-2*edge
            x_end   = x_start + batch_shape[2]
            z_start = 0
            z_end   = batch_shape[0]
            y_start = 0
            y_end   = batch_shape[1]
    
    
    if cell_dist==True:# Start of Cell centre location Analysis
        #calling cell_centre_distribution function from toolbox
        print ("starting analysis of cell centers")
        resultarray = cell_centre_distribution(bool_input=results[u"cellcenters"],reach=reach, sparsity_factor=sparsity_factor)

        if save == True:
            results[u"result_array"][:,:,:] = resultarray.astype("uint16")

        if plot == True:
            plt.figure(figsize=(8,8))
            plt.imshow(resultarray[resultarray.shape[0]//2,:,:],cmap="inferno")
            plt.colorbar()
            plt.show()
    
    if cell_dist==False:
        resultarray = np.zeros((1,1))# placeholder empty array
    
    #Saving of the resulting images and used variables
     
    if save == True:
       with open(save_path + r"/data" + timestamp + r".txt", "w") as variables:
           variables.write("name of image = " + str(img_path) + "\n")
           variables.write("batch shape = " + str(batch_shape[0]) + "," + str(batch_shape[1]) + "," + str(batch_shape[2]) + "\n")
           variables.write("gaussian sigma = " + str(gauss_sigma) + "\n")
           variables.write("gabor edge length = " + str(edge) + "\n")
           variables.write("gabor frequency = " + str(gabor_freq) + "\n")
           variables.write("gabor phase = " + str(gabor_phase) + "\n")
           variables.write("gabor sigma = " + str(gabor_sigma) + "\n")
           variables.write("gabor radius = " + str(gabor_radius) + "\n")
           variables.write("maxima detection radius = " + str(r_maxima) + "\n")
           variables.write("number of surrounding pixels taken into account per cell: " + str(reach) + "\n")
           variables.write("subsampling of cell centres factor: " + str(sparsity_factor) + "\n")
           variables.write("start time (ymd_h) = " + str(timestamp) + "\n")
           variables.write("time for analysis = " + str(int((time.time() - start_time)/60)) + "min / " + str(round(((time.time() - start_time)/3600),2)) + "h" + "\n")
           variables.write("number of cells = " + str(n_cells))
     
    #close all h5 files
    closeall()
    print ("script runtime: " + str(int((time.time() - start_time)/60)) + "min / " + str(round(((time.time() - start_time)/3600),2)) + "h")
    print ("Continuing with next file")

print ("All done!")

#shutdown
if shutdown == True:
    os.system('shutdown -s -t 300') #to abort accidental shutdown on windows: "shutdown /a"
#code graveyard
