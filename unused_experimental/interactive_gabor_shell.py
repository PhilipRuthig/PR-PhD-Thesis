import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import test_volume

fig = plt.figure()
ax = fig.add_subplot(111)
# Adjust the subplots region to leave some space for the sliders and buttons
fig.subplots_adjust(bottom=0.3)

radius_0 = 6
sigma_0 = 8
freq_0 = 0.1
phase_0 = 4.7124

# Draw the initial plot
# The 'line' variable is used for modifying the line later
img = ax.imshow(test_volume._gabor_shell(radius=radius_0, sigma=sigma_0, freq=freq_0, phase=phase_0)[26,:,:], cmap='gray')

# Add sliders for tweaking the parameters
radius_slider_ax = fig.add_axes([0.25, 0.25, 0.65, 0.03])
radius_slider = Slider(radius_slider_ax, 'Radius', 4, 20, valinit=radius_0)
freq_slider_ax = fig.add_axes([0.25, 0.20, 0.65, 0.03])
freq_slider = Slider(freq_slider_ax, 'Freq', 0.01, 1, valinit=freq_0)
sigma_slider_ax = fig.add_axes([0.25, 0.15, 0.65, 0.03])
sigma_slider = Slider(sigma_slider_ax, 'Sigma', 1, 20, valinit=sigma_0)
phase_slider_ax = fig.add_axes([0.25, 0.10, 0.65, 0.03])
phase_slider = Slider(phase_slider_ax, 'Phase', 0, 6.28, valinit=phase_0)

# Define an action for modifying the line when any slider's value changes
def sliders_on_changed(val):
	data = test_volume._gabor_shell(radius=radius_slider.val, sigma=sigma_slider.val, freq=freq_slider.val, phase=phase_slider.val)[26,:,:]
	data = data/data.max()
	img.set_data(data)
	fig.canvas.draw_idle()
radius_slider.on_changed(sliders_on_changed)
freq_slider.on_changed(sliders_on_changed)
sigma_slider.on_changed(sliders_on_changed)
phase_slider.on_changed(sliders_on_changed)

plt.show()
