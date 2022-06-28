import slab_microscopy as sm
import numpy as np
import matplotlib.pyplot as plt
import tifffile as tf
'''
Puts annotated cell centres into bins across cortical depth and plots them. Used for the OHBM 2020 Poster.
'''


#bin size
binsize=[167,960,60]

#open images
folder = r"F:\ohbm_fig\data"
list_img = [r"\AJ001_left_HuCD_Cellcenters.tif",r"\AJ001_left_ToPro_Cellcenters.tif",
            r"\AJ001_right_HuCD_Cellcenters.tif",r"\AJ001_right_ToPro_Cellcenters.tif"]

#initialize result lists
x = np.linspace(start=0,stop=(900*0.542),num=(900//binsize[2]))# x axis
y_hucd_left = []
y_hucd_right = []
y_topro_left = []
y_topro_right = []

#do analysis on cell 
for img in list_img:
    centres = tf.imread(folder+img).astype("bool")#open current img
    centres = centres[22:47,:,:]
    binsize[0]=centres.shape[0]
    if "right" in img:
        centres=np.flip(centres,axis=2)#flip x axis if right side
    gen = sm.chunk_generator(centres.shape,binsize,overlap=0)#initialize generator
    for i in range(900//binsize[2]):
        z1,z2,y1,y2,x1,x2=next(gen)#get chunk coordinates
        current_img = centres[z1:z2,y1:y2,x1:x2]#crop img
        current_y=np.count_nonzero(current_img)#get cellcount
        if "ToPro" in img:
            if "left" in img:
                y_topro_left.append(current_y*337.707)
            if "right" in img:
                y_topro_right.append(current_y*337.707)
        if "HuCD" in img:
            if "left" in img:
                y_hucd_left.append(current_y*337.707)
            if "right" in img:
                y_hucd_right.append(current_y*337.707)

print("done")

labels = ["Neurons","Glia"]
y_topro_left_corrected = np.array(y_topro_left) - np.array(y_hucd_left)
y_topro_right_corrected= np.array(y_topro_right)- np.array(y_hucd_right)

fig,ax = plt.subplots(nrows=1,ncols=2,sharex=True,sharey=True,figsize=(12,5),gridspec_kw={"wspace":0.04})
plt.xlim(0,480)
ax[0].plot(x,y_hucd_left,label=labels[0],color="crimson",linewidth=2.6)
ax[0].plot(x,y_topro_left_corrected,label=labels[1],color="deepskyblue",linewidth=2.6)
ax[0].grid(b=True,axis="y",alpha=0.9,linestyle="--")
ax[0].set_xlabel("Cortical depth [um]",size=18)
ax[0].set_ylabel("cell density [n/mm³]",size=18)
ax[0].tick_params(axis="both",which="major",labelsize=14)
leg = ax[0].legend(fontsize="14",loc="upper left")


ax[1].plot()
ax[1].plot(x,y_hucd_right,label=labels[0],color="crimson",linewidth=2.6)
ax[1].plot(x,y_topro_right_corrected,label=labels[1],color="deepskyblue",linewidth=2.6)
ax[1].grid(b=True,axis="y",alpha=0.9,linestyle="--")
ax[1].tick_params(length=0)
ax[1].tick_params(axis="both",which="major",labelsize=15)
plt.savefig(r"C:\Users\pr55gone\Desktop\leftright.png",dpi=200,bbox_inches="tight")
plt.show()


# left hemisphere

# y_topro_left_corrected=np.array(y_topro_left)-np.array(y_hucd_left)
# labels = ["Neurons","Glia"]
# fig = plt.plot(x,y_hucd_left,label=labels[0],color="crimson")
# plt.plot(x,y_topro_left_corrected,label=labels[1],color="deepskyblue")
# plt.grid(b=True,axis="y",alpha=0.9,linestyle="-.")
# plt.title("Left A1", fontsize=15, verticalalignment='bottom')
# ax[0].set_xlabel("Cortical depth [um]")
# ax[0].set_ylabel("cell density [n/mm³]")

# #right 
# y_topro_right_corrected=np.array(y_topro_right)-np.array(y_hucd_right)
# plt.plot(x,y_hucd_right,label=labels[0],color="crimson")
# plt.plot(x,y_topro_right_corrected,label=labels[1],color="deepskyblue")
# plt.legend(loc='upper right')
# plt.grid(b=True,axis="y",alpha=0.9,linestyle="-.")
# plt.title("Right A1", fontsize=15, verticalalignment='bottom')
# plt.savefig(r"C:\Users\pr55gone\Desktop\leftright.png",dpi=200,bbox_inches="tight")
# plt.show()

#### code graveyard


# # left hemisphere

# y_topro_left_corrected=np.array(y_topro_left)-np.array(y_hucd_left)

# y = np.vstack([y_hucd_left,y_topro_left_corrected])

# labels = ["Neurons","Glia"]

# fig, ax = plt.subplots()
# ax.stackplot(x, y_topro_left_corrected, y_hucd_left, labels=labels, colors=("crimson","deepskyblue"))
# plt.grid(b=True,axis="y",alpha=0.9,linestyle="-.")
# ax.set_title("Left A1", fontsize=15, verticalalignment='bottom')
# ax.set_xlabel("Cortical depth [um]")
# ax.set_ylabel("cell density [n/mm³]")
# plt.savefig(r"C:\Users\pr55gone\Desktop\left.png",dpi=200,bbox_inches="tight")
# plt.show()

# #right 
# y_topro_right_corrected=np.array(y_topro_right)-np.array(y_hucd_right)

# y = np.vstack([y_hucd_right,y_topro_right_corrected])

# labels = ["Neurons","Glia"]

# fig, ax = plt.subplots()
# ax.stackplot(x, y_topro_right_corrected, y_hucd_right, labels=labels, colors=("crimson","deepskyblue"))
# ax.legend(loc='upper right')
# plt.grid(b=True,axis="y",alpha=0.9,linestyle="-.")
# ax.set_title("Right A1", fontsize=15, verticalalignment='bottom')
# ax.set_xlabel("Cortical depth [um]")
# ax.set_ylabel("cell density [n/mm³]")
# plt.savefig(r"C:\Users\pr55gone\Desktop\right.png",dpi=200,bbox_inches="tight")
# plt.show()
