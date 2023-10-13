#!/usr/bin/env python
import pickle
import numpy as np
from astropy.io import fits
import Functions
import matplotlib.pyplot as plt
import skimage.io
import skimage.filters
import pymaster as nmt
import pysap
import pprint
from params import *

def plot_wavelet_undec(data, scale, wavelet='BsplineWaveletTransformATrousAlgorithm',data_g=0,threshold=0.04):
    import pysap
    if data_g.all() == 0:
        data_g = data
    transform_klass = pysap.load_transform(wavelet)
    transform = transform_klass(nb_scale=scale, verbose=1, padding_mode="symmetric")
    transform.data = data_g
    transform.analysis()
    fig, axs = plt.subplots(2,int(scale/2))
    for n in range(0,scale):
        if n < scale/2:
            axs[0,n].imshow(transform.analysis_data[n])
            axs[0,n].get_yaxis().set_visible(False)
            axs[0,n].get_xaxis().set_visible(False)
        else:
            axs[1,int(n-scale/2)].imshow(transform.analysis_data[n])
            axs[1,int(n-scale/2)].get_yaxis().set_visible(False)
            axs[1,int(n-scale/2)].get_xaxis().set_visible(False)
    fig.suptitle("Wavelet transform of n="+str(scale)+" scales"
                                                      " ("+str(wavelet)+")")
    plt.savefig('waveletundec_decomp_'+str(wavelet)+'_'+str(scale)+'.png')
    return transform.analysis_data

def get_fluctuations_wavelet_undec(data, scales, wavelet='BsplineWaveletTransformATrousAlgorithm', data_g=0, smoothlvl=1, noiselvl=0):
    import pysap
    if data_g.all() == 0:
        data_g = data
    transform_klass = pysap.load_transform(wavelet)
    transform = transform_klass(nb_scale=scales, verbose=1, padding_mode="symmetric")
    transform.data = data_g
    transform.analysis()
    # if smoothlvl == 1:
    #     smooth = transform.analysis_data[-smoothlvl]
    # else:
    smooth = np.sum(transform.analysis_data[-smoothlvl:],axis=0)
    print(np.shape(data))
    print(np.shape(smooth))
    y_fluc = data-smooth
    print(np.shape(y_fluc))
    if noiselvl != 0:
        transform1 = transform_klass(nb_scale=scales, verbose=1, padding_mode="symmetric")
        transform1.data = data
        transform1.analysis()
        # if noiselvl == 1:
        #     noise = transform.analysis_data[:noiselvl]
        # else:
        noise = np.sum(transform.analysis_data[:noiselvl],axis=0)
        y_fluc = y_fluc-noise
    y_fluc_norm = y_fluc/np.abs(smooth)

    return y_fluc, y_fluc_norm, smooth


y_fluc,y_fluc_norm,smooth = get_fluctuations_wavelet_undec(data1[279:419, 282:422], 6, wavelet='BsplineWaveletTransformATrousAlgorithm',data_g=data[279:419, 282:422],smoothlvl=2, noiselvl=1)
print(y_fluc.min())
print(y_fluc.max())
print(np.shape(y_fluc))
x_ticks = ['-2', '-1','0','1','2']
y_ticks = ['-2', '-1','0','1','2']
t11 = [0,35,70,105,138]
plt.figure()
norm = Functions.TwoSlopeNorm(vmin=y_fluc.min(), vcenter=0, vmax=y_fluc.max())
pc = plt.pcolormesh(y_fluc, norm=norm, cmap="seismic")
plt.imshow(y_fluc, cmap = 'seismic')
ax = plt.gca()
plt.savefig("y_fluc_first_bspline_6_2_1.png")
plt.close()


x_ticks = ['-2', '-1','0','1','2']
y_ticks = ['-2', '-1','0','1','2']
t11 = [0,35,70,105,138]
plt.figure()
norm = Functions.TwoSlopeNorm(vmin=-1.5, vcenter=0, vmax=1.5)
pc = plt.pcolormesh(y_fluc_norm, norm=norm, cmap="seismic")
plt.imshow(y_fluc_norm, cmap = 'seismic')
ax = plt.gca()
plt.savefig("y_fluc_norm_first_bspline_6_2_1.png")
plt.close()


plt.figure()
plt.imshow(smooth)
plt.savefig("y_smooth_bspline_6_2_1.png")
plt.close()
# plot_wavelet_undec(data1[279:419, 282:422], 6, wavelet='db3',data_g=data[279:419, 282:422])