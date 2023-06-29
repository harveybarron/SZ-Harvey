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

plot_wavelet_undec(data1[279:419, 282:422], 6, wavelet='db3',data_g=data[279:419, 282:422])