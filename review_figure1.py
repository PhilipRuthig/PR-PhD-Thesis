import slab
import matplotlib.pyplot as plt
'''
Generates Figure 1 in the Ruthig, Schönwiesner 2022 review.
'''

path_human=r'D:\MEGA\Uni_Zeugs\PhD\__Papers, Thesis\Review\Sound samples\Human_selfrecorded.wav'
path_marmoset=r'D:\MEGA\Uni_Zeugs\PhD\__Papers, Thesis\Review\Sound samples\Marmoset for Marc_cropped.wav'
path_mouse=r'D:\MEGA\Uni_Zeugs\PhD\__Papers, Thesis\Review\Sound samples\pone.0046610.s001.wav'

snd_human=slab.Sound(path_human)
snd_marmoset=slab.Sound(path_marmoset)
snd_mouse=slab.Sound(path_mouse)

snd_human.spectrogram(dyn_range=55)
snd_marmoset.spectrogram(dyn_range=55)

# pitch: compressed by factor 15. temporal. slowed by factor 15
snd_mouse.samplerate=snd_mouse.samplerate*15
snd_mouse.spectrogram(dyn_range=55)

# sound samples: 
# mouse: Arriaga et al., 2012
# marmoset: By Xiaoqin Wang Laboratory, Johns Hopkins Medicine (https://www.eurekalert.org/multimedia/pub/105940.php?from=315213)
# human: self recorded