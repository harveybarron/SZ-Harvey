import numpy as np
from astropy.io import fits
import Functions
import matplotlib.pyplot as plt
import skimage.io
import skimage.filters
from matplotlib.colors import TwoSlopeNorm

hudl = fits.open('ICM_old/data/map2048_MILCA_Coma_20deg_G.fits')
data = hudl[1].data

pixsize = 1.7177432059
#cen_x, cen_y = np.unravel_index(data.argmax(), data.shape)

rs, ys, step_size, maxval, ellipse = Functions.get_2Dys_in_annuli(data, [349.85768166,352.42445296,1,0],1.7177432059)

#rs, ys, step_size, maxval, ellipse = Functions.get_2Dys_in_annuli(data, [349,352,1,0])

# apply Gaussian blur, convolve with centred, gaussian vignette
blurred = skimage.filters.gaussian(data, sigma=(5, 5))
#plt.contour(blurred,np.arange(0,180,3))
#plt.imshow(blurred)
def gaussian_heatmap(center, image_size, sig = 1):
    """
    It produces single gaussian at expected center
    :param center:  the mean position (X, Y) - where high value expected
    :param image_size: The total image size (width, height)
    :param sig: The sigma value
    :return:
    """
    x_axis = np.linspace(0, image_size[0]-1, image_size[0]) - center[0]
    y_axis = np.linspace(0, image_size[1]-1, image_size[1]) - center[1]
    xx, yy = np.meshgrid(x_axis, y_axis)
    kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sig))
    return kernel

# contour lines
#kernel = gaussian_heatmap(center = (350, 350), image_size = (700, 700), sig = 50)
#plt.contour(blurred*kernel,np.linspace(0,np.max(data),10))

# fluctuations
y_fluc = data-blurred
y_fluc_norm = y_fluc / np.max(np.abs(blurred))

#plt.contour(ellipse*pixsize,np.arange(0,180,3))
#plt.imshow(np.log(data))


#fluctuations plot
x_ticks = ['-2', '-1','0','1','2']
y_ticks = ['-2', '-1','0','1','2']
t11 = [0,35,70,105,138]
plt.figure()
#plt.xticks(ticks=t11, labels=x_ticks, size='small')
#plt.yticks(ticks=t11, labels=y_ticks, size='small')
norm = TwoSlopeNorm(vmin=y_fluc_norm[279:419,282:422].min(), vcenter=0, vmax=y_fluc_norm[279:419,282:422].max())
pc = plt.pcolormesh(y_fluc_norm[279:419,282:422], norm=norm, cmap="seismic")
plt.imshow(y_fluc_norm[279:419,282:422], cmap = 'seismic')
ax = plt.gca()
plt.show()