import os
import SimpleITK as sitk
import tifffile as tf
'''
This script was meant to be used for registering an overview image to a mouse brain atlas dataset.
I can not recommend sITK for registration.
'''

fix_path = r"C:\Data\AJ003\results\best so far\step2_reference_image.tif"
mov_path = r"C:\Data\AJ003\results\best so far\step2_moving_image.tif"

#load images
fixed_image = sitk.GetImageFromArray(tf.imread(fix_path).astype("float32"))
moving_image = sitk.GetImageFromArray(tf.imread(mov_path).astype("float32"))

fixed_points = None
moving_points = None

registration_method = sitk.ImageRegistrationMethod()
# Create initial identity transformation.
transform_to_displacment_field_filter = sitk.TransformToDisplacementFieldFilter()
transform_to_displacment_field_filter.SetReferenceImage(fixed_image)
# The image returned from the initial_transform_filter is transferred to the transform and cleared out.
initial_transform = sitk.DisplacementFieldTransform(transform_to_displacment_field_filter.Execute(sitk.Transform()))

# Regularization (update field - viscous, total field - elastic).
initial_transform.SetSmoothingGaussianOnUpdate(varianceForUpdateField=0.0, varianceForTotalField=2.0) 

registration_method.SetInitialTransform(initial_transform)
registration_method.SetMetricAsDemons(10) #intensities are equal if the difference is less than 10HU
    
# Multi-resolution framework.
registration_method.SetShrinkFactorsPerLevel(shrinkFactors = [1,])
registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[1,])
registration_method.SetInterpolator(sitk.sitkLinear)

# If you have time, run this code as is, otherwise switch to the gradient descent optimizer    
#registration_method.SetOptimizerAsConjugateGradientLineSearch(learningRate=1.0, numberOfIterations=20, convergenceMinimumValue=1e-6, convergenceWindowSize=10)
registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=20, convergenceMinimumValue=1e-6, convergenceWindowSize=10)
registration_method.SetOptimizerScalesFromPhysicalShift()

# If corresponding points in the fixed and moving image are given then we display the similarity metric
# and the TRE during the registration.
if fixed_points and moving_points:
    registration_method.AddCommand(sitk.sitkStartEvent, rc.metric_and_reference_start_plot)
    registration_method.AddCommand(sitk.sitkEndEvent, rc.metric_and_reference_end_plot)        
    registration_method.AddCommand(sitk.sitkIterationEvent, lambda: rc.metric_and_reference_plot_values(registration_method, fixed_points, moving_points))
    
transformation = registration_method.Execute(fixed_image, moving_image) 

####
moving_rigid_transformed = sitk.Resample(moving_image, fixed_image, transformation, sitk.sitkLinear, 0.0, moving_image.GetPixelID())

# save results to file
tf.imsave(r"C:\Data\AJ003\results\best so far\demons\de_moving_image.tif", sitk.GetArrayFromImage(moving_rigid_transformed))
tf.imsave(r"C:\Data\AJ003\results\best so far\demons\de_moving_rigid_transformed_image.tif", sitk.GetArrayFromImage(moving_rigid_transformed))
tf.imsave(r"C:\Data\AJ003\results\best so far\demons\de_reference_image.tif", sitk.GetArrayFromImage(fixed_image))
sitk.WriteTransform(transformation, r"C:\Data\AJ003\results\best so far\demons\de_transform.tfm")
