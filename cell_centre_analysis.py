"""
@author: Philip Ruthig, PhD Student at Leipzig University & MPI for Cognitive and Brain Sciences.
This script replicates Figure 3.2.4.1.1 and Figure 3.2.8.1 in my Thesis. It performs the neccessary calculations on the previously
annotated and analyzed cell distributions from cell_annotation.py. 
"""
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
import matplotlib.gridspec as gridspec


def chunk_generator(img_shape,chunk_size,overlap):
    '''
    Returns a sequence of coordinates every time it is called with next() that can be used to cycle through 3D arrays in blocks.

    Inputs:
    img_shape: image shape(z,y,x)
    chunk_size: desired chunk size (z,y,x)
    overlap: overlap (in pixels) on every side of the chunk

    Outputs:
    6 integers giving the start & end coordinates in all axes in the following order:
    xstart, xend, ystart, yend, zstart, zend

    to do:
        rest of image calculation, uneven boundaries
        n-dimensional image compatibility
    '''

    z_start = 0 
    z_end   = chunk_size[0]
    y_start = 0
    y_end   = chunk_size[1]
    x_start = 0
    x_end   = chunk_size[2]
    
    while x_end <= img_shape[2]: #if x_end exceeds x boundary of image, all is done

        yield (z_start,z_end,y_start,y_end,x_start,x_end)

        z_start = z_start + chunk_size[0]-2*overlap
        z_end   = z_start + chunk_size[0]
        
        # if z_end exceeds img shape: move y_start (and reset z_start)
        if z_end > img_shape[0]:
            y_start = y_start + chunk_size[1]-2*overlap
            y_end   = y_start + chunk_size[1]
            z_start = 0
            z_end   = chunk_size[0]
        
        # if z_end AND y_end exceed img shape: move x_start (and reset y_start and z_start)
        if y_end > img_shape[1]:
            x_start = x_start + chunk_size[2]-2*overlap
            x_end   = x_start + chunk_size[2]
            z_start = 0
            z_end   = chunk_size[0]
            y_start = 0
            y_end   = chunk_size[1]

    yield z_start,z_end,y_start,y_end,x_start,x_end

# script options
sigma=2
vmin_single = -3.5
vmax_single = 3.5
vmin_avg=-1
vmax_avg=0.7#
subset=True
cutoff=1#pixels in Z to include in distance 1d plot

# img_folder = r"/data/ssd2/Philip/fourthrun_test/"
# img_paths = (
# r"results_PR009_l_ACx_20220512_09.58.h5",
# r"results_PR009_r_ACx_20220512_13.40.h5",
# # r"results_PR010_l_ACx_20220512_17.08.h5",
# # r"results_PR010_r_ACx_20220513_00.07.h5",
# r"results_PR012_l_ACx_20220513_04.49.h5",
# r"results_PR012_r_ACx_20220513_09.33.h5",
# r"results_PR014_l_ACx_20220513_17.04.h5",
# r"results_PR014_r_ACx_20220513_21.52.h5",
# # r"results_PR016_l_ACx_20220514_04.40.h5",
# # r"results_PR016_r_ACx_20220514_09.05.h5",
# r"results_PR017_l_ACx_20220514_17.48.h5",
# r"results_PR017_r_ACx_20220514_23.39.h5",
# r"results_PR018_l_ACx_20220515_07.02.h5",
# r"results_PR018_r_ACx_20220515_11.37.h5",)

img_folder = r"/data/ssd2/Philip/fifthrun_AC_only/"
#paths must be in L-R-L-R order
img_paths = (
# r"results_PR009_l_ACx_20220516_11.23.h5",
# r"results_PR009_r_ACx_20220516_13.34.h5",
r"results_PR010_l_ACx_20220518_16.51.h5",
r"results_PR010_r_ACx_20220518_22.11.h5",
r"results_PR012_l_ACx_20220516_15.49.h5",
r"results_PR012_r_ACx_20220516_22.22.h5",
r"results_PR014_l_ACx_20220517_02.49.h5",
r"results_PR014_r_ACx_20220517_09.48.h5",
r"results_PR017_l_ACx_20220517_15.53.h5",
r"results_PR017_r_ACx_20220517_20.29.h5",
r"results_PR018_l_ACx_20220519_03.15.h5",
r"results_PR018_r_ACx_20220519_11.29.h5",
r"results_cs002_l_ACx._20220530_16.53.h5",
r"results_cs002_r_ACx._20220530_17.45.h5",
r"results_cs003_l_ACx._20220530_18.26.h5",
r"results_cs003_r_ACx._20220530_19.14.h5",
r"results_cs004_l_ACx._20220530_19.49.h5",
r"results_cs004_r_ACx._20220530_20.39.h5",)


img_crops = []
# img_crops.append((20,106,40,1050,600,1800))#pr009_l
# img_crops.append((110,190,190,1350,110,1500))#pr009_r
img_crops.append((80,190,400,1600,80,1400))#pr010_l
img_crops.append((190,290,750,2100,20,1500))#pr010_r
img_crops.append((210,325,370,1800,430,1900))#pr012_l
img_crops.append((180,280,800,2000,400,1950))#pr012_r
img_crops.append((217,310,550,1800,250,2010))#pr014_l
img_crops.append((180,280,500,1800,130,2050))#pr014_r
img_crops.append((230,315,1000,2250,100,1900))#pr017_l
img_crops.append((195,290,600,1900,100,1800))#pr017_r
img_crops.append((70,180,550,1900,400,2000))#pr018_l
img_crops.append((80,185,550,1900,400,2000))#pr018_r
img_crops.append((338,450,110,2160,0,1300))#cs002_l
img_crops.append((486,612,100,1850,1260,2560))#cs002_r
img_crops.append((382,510,0,1299,300,2050))#cs003_l
img_crops.append((460,580,0,1300,360,2160))#cs003_r
img_crops.append((390,510,50,1350,40,1800))#cs004_l
img_crops.append((340,450,1260,2560,110,2050))#cs004_r

img_volumes=[]
for img_crop in img_crops:
    img_volumes.append((img_crop[1]-img_crop[0])*(img_crop[3]-img_crop[2])*(img_crop[5]-img_crop[4]))

avg_img_volumes_r = img_volumes[1::2]
avg_img_volumes_l = img_volumes[::2]

list_img=[]
list_img_labels=[]
list_lr_diffs=[]

for img_path in img_paths:
    # open h5
    img_file = h5py.File(img_folder + img_path, "r")
    temp_resultarray = img_file[u"result_array"]
    # rotate picture 90° right (left AC) or 270° right (right AC)
    if "l_ACx" in img_path:
        rot_temp_resultarray = np.rot90(temp_resultarray,k=3,axes=(1,2))
        list_img_labels.append(img_path[8:15])
    else:
        rot_temp_resultarray = np.rot90(temp_resultarray,k=1,axes=(1,2))
        list_img_labels.append(img_path[8:15])
    # append image to list
    list_img.append(rot_temp_resultarray)


#### swap images for PR014 and PR017, which were acquired with L/R orientation flipped
list_img[6], list_img[7] = list_img[7], list_img[6]
list_img[8], list_img[9] = list_img[9], list_img[8]
img_crops[6], img_crops[7] = img_crops[7], img_crops[6]
img_crops[8], img_crops[9] = img_crops[9], img_crops[8]

# blur images, z score and calculate averages. 
for i,img in enumerate(list_img):
    if i == 0:
        list_img_raw=[]
        avg_left = np.zeros_like(img,dtype="float32") #initialize empty arrays
        avg_right = np.zeros_like(img,dtype="float32")
        left_raw_sum = np.zeros_like(img,dtype="uint16")
        right_raw_sum = np.zeros_like(img,dtype="uint16")
    
    list_img_raw.append(img)

    if "_l" in list_img_labels[i]: #add all data by hemisphere, make average
        left_raw_sum+=img
    else:
        right_raw_sum+=img
    
    img = img.astype("float32")
    img = ndi.filters.gaussian_filter(img, sigma=sigma)
    img = img - np.mean(img)
    img = img / np.std(img)
    list_img[i] = img

    if "_l" in list_img_labels[i]: #add all data by hemisphere, make average
        avg_left+=img
    else:
        avg_right+=img
    
avg_left=avg_left/i/2 #divide by number of images in data 
avg_right=avg_right/i/2
avg_diff=avg_left-avg_right #calculate average difference left-right

#
# Fig1
#
# calculate differences between each left and right AC image
for i, img in enumerate(list_img):
    if "_l" in list_img_labels[i]: # calculate difference between left and right hemisphere of single brains
        img_diff = list_img[i]-list_img[i+1] # left minus right hemisphere
    else:
        continue
    list_lr_diffs.append(img_diff) 

# plotting l/r differences and averages
fig,ax=plt.subplots(nrows=len(img_paths)//2+1,ncols=3,sharex="all",sharey="all",figsize=(6,24))
for i, img in enumerate(list_img):
    if "_l" in list_img_labels[i]:
        ax[i//2,0].imshow(list_img[i][avg_left.shape[0]//2,:,:],vmin=vmin_single,vmax=vmax_single)#left
        temp = list_img[i][avg_left.shape[0]//2,:,:]#store current left hemisphere img in temp for difference calculation
    if "_r" in list_img_labels[i]:
        ax[i//2,1].imshow(list_img[i][avg_left.shape[0]//2,:,:],vmin=vmin_single,vmax=vmax_single)#right
        ax[i//2,2].imshow(temp-list_img[i][avg_left.shape[0]//2,:,:],vmin=vmin_single,vmax=vmax_single)#diff
ax[len(img_paths)//2,0].imshow(avg_left[avg_left.shape[0]//2,:,:],vmin=vmin_avg,vmax=vmax_avg)
ax[len(img_paths)//2,1].imshow(avg_right[avg_right.shape[0]//2,:,:],vmin=vmin_avg,vmax=vmax_avg)
ax[len(img_paths)//2,2].imshow(avg_left[avg_left.shape[0]//2,:,:]-avg_right[avg_right.shape[0]//2,:,:],vmin=vmin_avg,vmax=vmax_avg)
fig.tight_layout()
plt.show()

#
# Fig2
#

# calculate 1D array of distances to the middle cell

if subset == True: #crop images to subset
    right_raw_sum=right_raw_sum[right_raw_sum.shape[0]//2-cutoff:right_raw_sum.shape[0]//2+cutoff]
    left_raw_sum=left_raw_sum[left_raw_sum.shape[0]//2-cutoff:left_raw_sum.shape[0]//2+cutoff]
    for i, img_temp in enumerate(list_img_raw):
        list_img_raw[i]=img_temp[img_temp.shape[0]//2-cutoff:img_temp.shape[0]//2+cutoff]

coords = chunk_generator((right_raw_sum.shape),chunk_size=(1,1,1),overlap=0,)
midpoint=[right_raw_sum.shape[0]//2,right_raw_sum.shape[1]//2,right_raw_sum.shape[2]//2]
list_distance_right = np.zeros(int(m.sqrt((right_raw_sum.shape[0]//2)**2 + (right_raw_sum.shape[1]//2)**2 + (right_raw_sum.shape[2]//2)**2))+1,)
list_distance_left = np.zeros(int(m.sqrt((right_raw_sum.shape[0]//2)**2 + (right_raw_sum.shape[1]//2)**2 + (right_raw_sum.shape[2]//2)**2))+1,)

# initialize empty list and image filled with ones for normalization
list_normalization = np.zeros(int(m.sqrt((right_raw_sum.shape[0]//2)**2 + (right_raw_sum.shape[1]//2)**2 + (right_raw_sum.shape[2]//2)**2))+1,)
img_normalization = np.ones_like(right_raw_sum)

for i in range(right_raw_sum.shape[0]*right_raw_sum.shape[1]*right_raw_sum.shape[2]):
    #set active pixel
    coord_tuple=next(coords)
    coords_active=[]
    coords_active.append(coord_tuple[0])
    coords_active.append(coord_tuple[2])
    coords_active.append(coord_tuple[4])
    # calculate distance from mid pixel to active pixel
    current_dist = m.sqrt((coords_active[0]-midpoint[0])**2 + (coords_active[1]-midpoint[1])**2 + (coords_active[2]-midpoint[2])**2)
    # readout number of cells in active pixel
    n_cells_active_right = right_raw_sum[coords_active[0],coords_active[1],coords_active[2]]
    n_cells_active_left = left_raw_sum[coords_active[0],coords_active[1],coords_active[2]]
    # add the number of active cells in the raw sum array into the list
    list_normalization[int(current_dist)]+=1
    list_distance_right[int(current_dist)]+=n_cells_active_right
    list_distance_left[int(current_dist)]+=n_cells_active_left

# normalize by amount of (possible) cell locations
list_distance_right = list_distance_right/list_normalization
list_distance_left = list_distance_left/list_normalization
# remove normalization artifact
list_distance_left[0]=0 
list_distance_right[0]=0 

single_distance_array = np.zeros((len(img_paths),len(list_distance_left)+1))

for i,current_img in enumerate(list_img_raw):
    # calculate 1d distance plots for every dist array for standard deviation
    coords = chunk_generator((list_img_raw[0].shape),chunk_size=(1,1,1),overlap=0,)
    for px in range(current_img.shape[0]*current_img.shape[1]*current_img.shape[2]):
        #set active pixel
        coord_tuple=next(coords)
        coords_active=[]
        coords_active.append(coord_tuple[0])
        coords_active.append(coord_tuple[2])
        coords_active.append(coord_tuple[4])
        # calculate distance from mid pixel to active pixel
        current_dist = m.sqrt((coords_active[0]-midpoint[0])**2 + (coords_active[1]-midpoint[1])**2 + (coords_active[2]-midpoint[2])**2)
        # readout number of cells in active pixel
        n_cells_temp = current_img[coords_active[0],coords_active[1],coords_active[2]]
        # add the number of active cells in the raw sum array into the list
        single_distance_array[i,int(current_dist)]+=n_cells_temp
        # normalize by amount of possible cell locations
        
    single_distance_array[i,:-1] = single_distance_array[i,:-1]/list_normalization
    single_distance_array[:,0] = 0

list_std_left = [] 
list_std_right = []
for i,distance in enumerate(range(single_distance_array.shape[1])):
    list_std_left.append(np.std(single_distance_array[0:10:2,i]))
    list_std_right.append(np.std(single_distance_array[1:10:2,i]))

# plot with errorbars
fig = plt.figure(figsize=(8, 4))
ax = fig.add_subplot(2, 1, 2,)
plt.errorbar(np.array(range(len(list_distance_right[:60])))*0.542,list_distance_right[:60],yerr=list_std_right[:60],elinewidth=1.2,color="goldenrod",label="right AC",alpha=0.7,lw=2)#replace error bars
plt.errorbar(np.array(range(len(list_distance_left[:60])))*0.542,list_distance_left[:60],yerr=list_std_left[:60],elinewidth=1.2,color="blue",label="left AC",alpha=0.7,lw=2)
plt.legend(loc="lower right")
ax.set_xlabel("distance from reference cell [µm]")
ax.set_ylabel("average number of cells per voxel at radius x")
plt.show()

# # calculate average distance +-std for all cells
# temp_l, temp_r = 0,0
# std_list_l=[0]*len(list_distance_left) # initialize list of zeros for std calculation
# std_list_r=[0]*len(list_distance_right) 
# for i,dist in enumerate(list_distance_left):#sum up all elements in distance list
#     temp_l += i*list_distance_left[i]
#     temp_r += i*list_distance_right[i]
#     std_list_l[i] = list_distance_left[i]*i
#     std_list_r[i] = list_distance_right[i]*i
# # average
# temp_l = temp_l/i
# temp_l_transp = [0]*int(np.max(std_list_l))
# temp_r = temp_r/i
# temp_r_transp = [0]*int(np.max(std_list_r))


# # std_l = np.std(std_list_l)
# # std_r = np.std(std_list_r)

# avg_left = 
# avg_right = 

# print ("average left distance " + avg_left + "µm +-" + str(std_l))
# print ("average right distance " + avg_right + "µm +-"+ str(std_r))

# save all as tiff
for i, img_path in enumerate(img_paths):
    # save as TIFF
    tf.imwrite(img_folder + r"/processed/" + list_img_labels[i] + r".tif", list_img[i])
    tf.imwrite(img_folder + r"/processed/" + list_img_labels[i] + r".tif", list_img[i])
    tf.imwrite(img_folder + r"/processed/" + list_img_labels[i] + r"_raw.tif", list_img_raw[i])
    if i % 2 == 0 and i != 0:
        tf.imwrite(img_folder + r"/processed/" + list_img_labels[i] + r"_raw.tif", list_lr_diffs[i//2])

tf.imwrite(img_folder + r"/processed/left_avg.tif", avg_left)
tf.imwrite(img_folder + r"/processed/right_avg.tif", avg_right)
tf.imwrite(img_folder + r"/processed/diff_avg.tif", avg_left-avg_right)

print("done")

#
# Fig3
#

#initialize lists of cells
list_n_cells_l = [] 
list_n_cells_r = []

for img_path in img_paths:
    # open h5
    print (img_path)
    img_file = h5py.File(img_folder + img_path, "r")
    temp_resultarray = img_file[u"cellcenters"]
    if "l_ACx" in img_path:
        list_n_cells_l.append(np.sum(temp_resultarray,axis=(0,1,2)))
    else:
        list_n_cells_r.append(np.sum(temp_resultarray,axis=(0,1,2)))

avg_left = np.average(list_n_cells_l)/np.average(avg_img_volumes_l)*567348846.92*1.398#factor for correction to density to cells/mm and factor in non-tissue areas in images
avg_right = np.average(list_n_cells_r)/np.average(avg_img_volumes_r)*567348846.92*1.398
labels = ["left AC", "right AC"]

densities_l=np.array((list_n_cells_l))/np.array((avg_img_volumes_l))*567348846.92*1.398
densities_r=np.array((list_n_cells_r))/np.array((avg_img_volumes_r))*567348846.92*1.398

fig,ax = plt.subplots()
ax.bar(labels, (avg_left,avg_right),color=("blue","goldenrod"),alpha=0.7,width=0.65,ecolor="black")
ax.yaxis.grid(True)
ax.set_ylabel("density of cells [n/mm³]")
ax.set_ylim(0,90000)
for i in range(len(densities_l)):
    plt.plot((densities_l[i],densities_r[i]),marker=".",color="black")
plt.show()

### code graveryard

# average over left & right AC images
# for i,img in enumerate(list_img):
#     if "_l" in list_img_labels[i]: #add all data by hemisphere
#         avg_left+=img
#     else:
#         avg_right+=img

# img_folder = r"/data/ssd2/Philip/first_run/"
# img_paths = (
# r"results_PR009_l_ACx_20220415_15.27.h5",
# r"results_PR009_r_ACx_20220415_17.25.h5",
# r"results_PR010_l_ACx_20220415_19.18.h5",
# r"results_PR010_r_ACx_20220415_23.30.h5",
# r"results_PR012_l_ACx_20220416_02.18.h5",
# r"results_PR012_r_ACx_20220416_04.57.h5",
# r"results_PR014_l_ACx_20220416_08.32.h5",
# r"results_PR014_r_ACx_20220416_11.14.h5",
# r"results_PR016_l_ACx_20220416_14.47.h5",
# r"results_PR016_r_ACx_20220416_16.46.h5",
# r"results_PR017_l_ACx_20220416_21.02.h5",
# r"results_PR017_r_ACx_20220417_00.21.h5",
# r"results_PR018_l_ACx_20220417_04.11.h5",
# r"results_PR018_r_ACx_20220417_06.40.h5",)

# img_folder = r"/data/ssd2/Philip/secondfullrun/"
# img_paths = (
# r"results_PR009_l_ACx_20220504_11.44.h5",
# r"results_PR009_r_ACx_20220504_15.09.h5",
# r"results_PR010_l_ACx_20220504_18.13.h5",
# r"results_PR010_r_ACx_20220505_01.42.h5",
# r"results_PR012_l_ACx_20220505_06.37.h5",
# r"results_PR012_r_ACx_20220505_11.10.h5",
# r"results_PR014_l_ACx_20220505_17.33.h5",
# r"results_PR014_r_ACx_20220505_22.12.h5",
# r"results_PR016_l_ACx_20220506_04.32.h5",
# r"results_PR016_r_ACx_20220506_07.50.h5",
# r"results_PR017_l_ACx_20220506_15.43.h5",
# r"results_PR017_r_ACx_20220506_21.20.h5",
# r"results_PR018_l_ACx_20220507_03.54.h5",
# r"results_PR018_r_ACx_20220507_07.59.h5",)