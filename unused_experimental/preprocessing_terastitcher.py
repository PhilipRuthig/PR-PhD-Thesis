#
'''
Concatenates LaVision Ultramicroscope II data in Z in multiple channels and in mosaics. 
outputs correct file format for Terastitcher Stitching
So far only works for Data generated with 12x objective! 

only works for images in range smaller than 10000µm 
'''

import os
import tifffile as tf
import numpy as np

# microscope parameters - need tweaking according to microscopy data
overlap = 0.10 # tile overlap in %
x_tile_microns = 1170 # image Dimensions of one image in µm. 1170x1387 for 20x objective
y_tile_microns = 1387
n_tiles_x, n_tiles_y = 0,2 #define number of tiles. ultraII file handle format: [XX x YY]
n_channels = 3 #number of channels to be concatenated

# script options 
input_path = r"G:\IDISCO_20200407_CS002_Detail\200407_cS002_allchannel_16-34-10" #path to raw files 
raw_filename = r"\16-34-10_cS002_allchannel_UltraII" #raw filename (before Z stack and mosaic coordinates)
output_path = r"F:\Philip\claudius_crops" #parent directory for output file structure

channels = (r"C02",) #list of channels to be processed (as given in the filenames)

final_shape = (801,2560,2160) #final shape of a single concatenated tile (z,y,x)

# initialize variables
current_x = 0
current_y = 0
n_iter = 0
x_tile_offset = int(x_tile_microns - (x_tile_microns*overlap)*2) # calculate offset in µm (minus overlap)
y_tile_offset = int(y_tile_microns - (y_tile_microns*overlap)*2)

while current_x <= n_tiles_x: #iterate through XY tiles
# while n_iter < 1:

    n_iter += 1
    
    #multiply by 10 to get to tenths of µm format needed for terastitcher
    x_dist = str(current_x * x_tile_offset * 10) 
    y_dist = str(current_y * y_tile_offset * 10)
    
    #get x_dist and y_dist to 6 digits
    while len(x_dist) < 6:
        x_dist = "0" + x_dist
    
    while len(y_dist) < 6:
        y_dist = "0" + y_dist

    current_path = output_path + "\\" + str(x_dist)
    if os.path.isdir(current_path) == False:
        os.mkdir(current_path)

    current_path += "\\" + str(x_dist) + "_" + str(y_dist)
    if os.path.isdir(current_path) == False:
        os.mkdir(current_path)

    stack = np.zeros(final_shape, dtype=np.uint16) #initialize empty image

    print ("concatenating channels for mosaic tile Y=" + str(current_y) + " X=" + str(current_x)) #concatenate all channels for given tile

    #define current tile in file handle
    prefix = raw_filename + r"[0" + str(current_y) + r" x 0" + str(current_x) + r"]_" #prefix of filename prefix (before z index)
    suffix = r".tif" #suffix of filename (after z index)

    for channel in range(len(channels)): #iterate through channels

        prefix_channel = prefix + channels[channel] + r"_xyz-Table Z" #adding channel to filename prefix

        print ("starting concatenating for " + channels[channel])

        for z in range(final_shape[0]): #iterate through z

            z_number = (4-len(str(z)))*"0" + str(z) #generate four-digit Z number
                        
            if z % 100 == 0: #timekeeping
                print ("active Channel: " + channels[channel] + " -- active z slice: " + str(z))

            current_z = tf.imread(input_path + prefix_channel + z_number + suffix, multifile = False) #adding Z index to filename and opening the relevant file
            stack[z,:,:] = current_z
        
        print ("Done with " + channels[channel] + "!")

    stack = stack.astype(r"uint16")
    tf.imsave(file=current_path + r"\000000.tif", data=stack, bigtiff=True, dtype="uint16", planarconfig="CONTIG") #save as tiff

    current_y +=1

    if current_y > n_tiles_y:
        current_y = 0
        current_x +=1
        continue

    else:
        print ("Saving File")


#code graveyard

# if n_iter == 1: #only first time: read metadata from file
#     first_file = tf.TiffFile(input_path + prefix_channel + z_number + suffix, multifile = False)
#     metadata = first_file.ome_metadata
#     str_metadata = metadata["Image"]["CustomAttributes"]["TileConfiguration"]["TileConfiguration"] #tileconfiguration str