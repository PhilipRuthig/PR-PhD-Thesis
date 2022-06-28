import os
import SimpleITK as sitk
from matplotlib import pyplot as plt
import tifffile as tf
from ipywidgets import interact, fixed
from IPython.display import clear_output
'''
Experimental rigid regitration pipeline using sITK. Don't use sITK.
'''


# Callback invoked by the interact IPython method for scrolling through the image stacks of
# the two images (moving and fixed).
def display_images(fixed_image_z, moving_image_z, fixed_npa, moving_npa):
    # Create a figure with two subplots and the specified size.
    plt.subplots(1,2,figsize=(10,8))
    
    # Draw the fixed image in the first subplot.
    plt.subplot(1,2,1)
    plt.imshow(fixed_npa[fixed_image_z,:,:],cmap="gray")
    plt.title('fixed image')
    plt.axis('off')
    
    # Draw the moving image in the second subplot.
    plt.subplot(1,2,2)
    plt.imshow(moving_npa[moving_image_z,:,:],cmap="gray")
    plt.title('moving image')
    plt.axis('off')
    
    plt.show()

# Callback invoked by the IPython interact method for scrolling and modifying the alpha blending
# of an image stack of two images that occupy the same physical space. 
def display_images_with_alpha(image_z, alpha, fixed, moving):
    img = (1.0 - alpha)*fixed[:,:,image_z] + alpha*moving[:,:,image_z] 
    plt.imshow(sitk.GetArrayViewFromImage(img),cmap="gray")
    plt.axis('off')
    plt.show()
    
# Callback invoked when the StartEvent happens, sets up our new data.
def start_plot():
    global metric_values, multires_iterations
    
    metric_values = []
    multires_iterations = []

# Callback invoked when the EndEvent happens, do cleanup of data and figure.
def end_plot():
    global metric_values, multires_iterations
    
    del metric_values
    del multires_iterations
    # Close figure, we don't want to get a duplicate of the plot latter on.
    plt.close()

# Callback invoked when the IterationEvent happens, update our data and display new figure.    
def plot_values(registration_method):
    global metric_values, multires_iterations
    
    metric_values.append(registration_method.GetMetricValue())                                       
    # Clear the output area (wait=True, to reduce flickering), and plot current data
    clear_output(wait=True)
    # Plot the similarity metric values
    plt.plot(metric_values, 'r')
    plt.plot(multires_iterations, [metric_values[index] for index in multires_iterations], 'b*')
    plt.xlabel('Iteration Number',fontsize=12)
    plt.ylabel('Metric Value',fontsize=12)
    plt.show()
    
# Callback invoked when the sitkMultiResolutionIterationEvent happens, update the index into the 
# metric_values list. 
def update_multires_iterations():
    global metric_values, multires_iterations
    multires_iterations.append(len(metric_values))        

mov_path = r"C:\Users\Philip\Desktop\Registration\HuCD_downscaled.tif"
fix_path = r"C:\Users\Philip\Desktop\Registration\autofluorescence_template.tif"

#load images
fixed_image = tf.imread(fix_path)
moving_image = tf.imread(mov_path)

#convert images to sitk format
fixed_image = sitk.GetImageFromArray(fixed_image)
moving_image = sitk.GetImageFromArray(moving_image)

# alternative transform method: sitk.AffineTransform(3)
initial_transform = sitk.CenteredTransformInitializer(fixed_image, 
                                                      moving_image, 
                                                      sitk.Euler3DTransform(), 
                                                      sitk.CenteredTransformInitializerFilter.GEOMETRY)

moving_rigid_transformed = sitk.Resample(moving_image, fixed_image, initial_transform, sitk.sitkLinear, 0.0, moving_image.GetPixelID())

# registration routine

# initialize
registration_method = sitk.ImageRegistrationMethod()

# Similarity metric settings.
registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=2000)#default:100
registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
registration_method.SetMetricSamplingPercentage(0.5)#default: 0.01

registration_method.SetInterpolator(sitk.sitkLinear)

# Optimizer settings. defaults: learningRate=1.0, numberOfIterations=100, convergenceMinimumValue=1e-6, convergenceWindowSize=10 
registration_method.SetOptimizerAsGradientDescent(learningRate=0.5, numberOfIterations=100, convergenceMinimumValue=1e-6, convergenceWindowSize=10)
registration_method.SetOptimizerScalesFromPhysicalShift()

# Setup for the multi-resolution framework.            
registration_method.SetShrinkFactorsPerLevel(shrinkFactors = [4,2,1])
registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2,1,0])
registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

# Don't optimize in-place, we would possibly like to run this cell multiple times.
registration_method.SetInitialTransform(initial_transform, inPlace=False)

# Connect all of the observers so that we can perform plotting during registration.
registration_method.AddCommand(sitk.sitkStartEvent, start_plot)
registration_method.AddCommand(sitk.sitkEndEvent, end_plot)
registration_method.AddCommand(sitk.sitkMultiResolutionIterationEvent, update_multires_iterations) 
registration_method.AddCommand(sitk.sitkIterationEvent, lambda: plot_values(registration_method))

final_transform = registration_method.Execute(sitk.Cast(fixed_image, sitk.sitkFloat32), 
                                              sitk.Cast(moving_image, sitk.sitkFloat32))

# apply transform to moving image
moving_rigid_transformed = sitk.Resample(moving_image, fixed_image, final_transform, sitk.sitkLinear, 0.0, moving_image.GetPixelID())

# show aligned images
# Create a figure with three subplots and the specified size.
plt.subplots(2,2,figsize=(12,10))

half_z = int(sitk.GetArrayFromImage(fixed_image).shape[0]//2)

# Draw the fixed image in the first subplot.
plt.subplot(2,2,1)
plt.imshow(sitk.GetArrayFromImage(fixed_image)[half_z,:,:],cmap="gray")
plt.title('fixed image')
plt.axis('off')

# Draw the moving image in the second subplot.
plt.subplot(2,2,2)
plt.imshow(sitk.GetArrayFromImage(moving_image)[half_z,:,:],cmap="gray")
plt.title('moving image before alignment')
plt.axis('off')

# Draw the moving image in the third subplot.
plt.subplot(2,2,3)
plt.imshow(sitk.GetArrayFromImage(moving_rigid_transformed)[half_z,:,:],cmap="gray")
plt.title('moving image after alignment')
plt.axis('off')

# add both images together and show in fourth subplot.
compound_img = sitk.GetArrayFromImage(moving_rigid_transformed) + sitk.GetArrayFromImage(fixed_image)
plt.subplot(2,2,4)
plt.imshow(compound_img[half_z,:,:],cmap="gray")
plt.title('moving image + fixed image')
plt.axis('off')

plt.show()

# save results to file
tf.imsave(r"C:\Users\Philip\Desktop\Registration\results\rig_raw_moving_image.tif", sitk.GetArrayFromImage(moving_rigid_transformed))
tf.imsave(r"C:\Users\Philip\Desktop\Registration\results\rig_moving_transformed_image.tif", sitk.GetArrayFromImage(moving_rigid_transformed))
tf.imsave(r"C:\Users\Philip\Desktop\Registration\results\rig_reference_image.tif", sitk.GetArrayFromImage(fixed_image))
sitk.WriteTransform(final_transform, r"C:\Users\Philip\Desktop\Registration\results\rig_transform.tfm")

## 
## flexible transformation - flexible transform 25µm reference image to the (previously rigid aligned) light sheet data.
## After that: transform atlas image in the same way reference image was transformed
## 