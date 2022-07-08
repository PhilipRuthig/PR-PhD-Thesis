import numpy
import matplotlib.pyplot as plt
import volumes
import tifffile as tf
'''
This script is a proof-of-principle to detect different cell radii with the gabor annulus filter.
'''


radii = numpy.linspace(3, 12, 10).astype(int)
img, centers = volumes.dummy(radii)
# img = volumes.Volume(r'F:\Philip\AJ001\alex_crops\cropped\AJ001_Au1_left_HuCD_cropped.tif')
img.upsample_z(factor=3)
img.zscore()
img.clamp(0, 2, method='sigmoid') # clamp values between 0 and 3 std
img.blur(sigma=0.1, inplace=True)
stack = img.apply_gabor_filters(radii) #stack = list of filtered images
img_max = numpy.max(stack, axis=3) # collapse by maximum
img_max = volumes.Volume(img_max)
img_max.blur(sigma=2, inplace=True)
centers, _ = img_max.detect_blobs(r_min=numpy.min(radii), r_max=numpy.max(radii), method='peak')
print(centers)
result_array = numpy.empty((len(centers), len(radii)))  #creating an empty array
for i_center, center in enumerate(centers):
	result_array[i_center,:] = stack[center[0], center[1], center[2], :]
index_max = numpy.argmax(result_array, axis=1)
center_radii = radii[index_max]
print(center_radii)
# plt.matshow(result_array.T)
# plt.ylabel('Radius')
# plt.colorbar()
# plt.show()

detected_volume = volumes.result_volume(centers, center_radii, img)
img.plot(overlay=detected_volume)
# plt.figure()
# y = result_array[3,:]
# plt.bar(gabor_radius, height=y)
# plt.xlabel('Gabor-Radius')
# plt.ylabel('Intensität')
