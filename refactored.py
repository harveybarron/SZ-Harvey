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
from params import *
def compute_shit():
    # inputs[k] = y_fluc, y_fluc_norm, y_smooth
    inputsfirst={}
    inputslast={}
    #output[k] = [amp, std_amp, cl00_uncoupled, covar, ells_uncoupled]
    outputs={}


    # Fit ellipltical model to the data
    for k,v in emp.items():
        inputsfirst[k] = Functions.get_fluctuations_ellipse(data1, v[0], v[1], v[2], v[3], v[4], data_g=data)
        inputslast[k] = Functions.get_fluctuations_ellipse(data2, v[0], v[1], v[2], v[3], v[4], data_g=data)
        
    # Perform Gaussian smoothing on the data
    for k in gaussian.keys():
        inputsfirst[k] = Functions.get_fluctuations_gaussian(data1[279:419,282:422],gaussian[k],data_g=data[279:419,282:422])
        inputslast[k] = Functions.get_fluctuations_gaussian(data2[279:419, 282:422], gaussian[k], data_g = data[279:419,282:422])

    # Perform wavelet smoothing on the data
    for k in wavelet.keys():
        inputsfirst[k] = Functions.get_fluctuations_wavelet(data1[279:419,282:422], wavelet[k][1],wavelet=wavelet[k][0],threshold=wavelet[k][2],data_g=data[279:419,282:422])
        inputslast[k] = Functions.get_fluctuations_wavelet(data2[279:419, 282:422], wavelet[k][1], wavelet=wavelet[k][0],threshold=wavelet[k][2],data_g=data[279:419,282:422])
        Functions.plot_wavelet_coeff(data1[279:419, 282:422], wavelet[k][1],
                                                            wavelet=wavelet[k][0], threshold=wavelet[k][2],
                                                            data_g=data[279:419, 282:422])

    # Use nfw model
    for k in nfw.keys():
        inputsfirst[k] = Functions.get_fluctuations_nfw(data1[279:419,282:422],datanfw[279:419,282:422],data_g=data[279:419,282:422])
        inputslast[k] = Functions.get_fluctuations_nfw(data2[279:419,282:422],datanfw[279:419,282:422],data_g=data[279:419,282:422])

    # Compute the namaster spectrum
    for k in inputsfirst.keys():
        print(k)
        outputs[k] = Functions.namaster_spectrum_khatri(inputsfirst[k][0], namp[0],
                namp[1],namp[2],namp[3],namp[4],namp[5],namp[6],norm_y_fluc1=inputslast[k][0])
    pickle.dump(inputsfirst, open('inputsfirst.p', 'wb'))
    pickle.dump(inputsfirst, open('inputslast.p', 'wb'))
    pickle.dump(outputs, open('outputs.p', 'wb'))
    return inputsfirst,inputslast,outputs

x=input("(R)ecompute? ")

if x == 'R':
    inputsfirst,inputslast,outputs=compute_shit()
else:
    #inputs=np.load('inputs.npy', allow_pickle=True)
    #outputs=np.load("spectra.npy", allow_pickle=True)
    inputsfirst = pickle.load(open('inputsfirst.p', 'rb'))
    inputslast = pickle.load(open('inputslast.p', 'rb'))
    outputs = pickle.load(open('outputs.p', 'rb'))

rmsvals=[]
scales=[]
for k in gaussian.keys():
    scales.append(gaussian[k])
    rmsvals.append(np.sqrt(np.mean(inputsfirst[k][0]**2)))
print(scales)
print(rmsvals)

#Plot the spectrum
plt.figure()
plt.xscale('log')
cmap = plt.get_cmap('tab10')
colour = 0
for k in wavelet.keys():
        lambdas_inv = outputs[k][4]/ell_kpc
        amp=outputs[k][0]
        std_amp=outputs[k][1]
        # Multiply by 1e6 to compare with Khatri
        #plt.errorbar(lambdas_inv,amp*1e6, yerr=std_amp*1e6, fmt='b.', ecolor=cmap(colour),elinewidth=1,capsize = 4,label="sigma=6")
        plt.plot(lambdas_inv,amp*1e6,color=cmap(colour),label="w="+str(wavelet[k][0])+", depth="+str(wavelet[k][1])+", t="+str(wavelet[k][2]))
        #plt.plot(lambdas_inv, amp * 1e6, color=cmap(colour),label="sigma=" + str(gaussian[k]))
        colour += 1
plt.errorbar(khatrix,khatriy,yerr=(khatrierror), fmt='r.',ecolor='black',elinewidth=1,capsize = 4)
plt.xlabel("$1/\lambda$ ($kpc^{-1}$)")
plt.ylabel("$A_{\delta y} = (k^2P_{\delta y}(k)/(2\pi))^{1/2}(10^{-6})$")
plt.legend()
plt.title("Cross Power Spectrum of varying Wavelets")
plt.savefig('crosspowerspectrum_wavelets_khatri2.png')


# for k in inputsfirst.keys():
#     x_ticks = ['-2', '-1','0','1','2']
#     y_ticks = ['-2', '-1','0','1','2']
#     t11 = [0,35,70,105,138]
#     plt.figure()
#     #plt.xticks(ticks=t11, labels=x_ticks, size='small')
#     #plt.yticks(ticks=t11, labels=y_ticks, size='small')
#     norm = Functions.TwoSlopeNorm(vmin=inputsfirst[k][0].min(), vcenter=0, vmax=inputsfirst[k][0].max())
#     pc = plt.pcolormesh(inputsfirst[k][0], norm=norm, cmap="seismic")
#     plt.imshow(inputsfirst[k][0], cmap = 'seismic')
#     ax = plt.gca()
#     plt.savefig("y_fluc_first_"+k+".png")
#     plt.close()
#
# for k in inputsfirst.keys():
#     x_ticks = ['-2', '-1','0','1','2']
#     y_ticks = ['-2', '-1','0','1','2']
#     t11 = [0,35,70,105,138]
#     plt.figure()
#     #plt.xticks(ticks=t11, labels=x_ticks, size='small')
#     #plt.yticks(ticks=t11, labels=y_ticks, size='small')
#     norm = Functions.TwoSlopeNorm(vmin=-1.5, vcenter=0, vmax=1.5)
#     pc = plt.pcolormesh(inputsfirst[k][1], norm=norm, cmap="seismic")
#     plt.imshow(inputsfirst[k][1], cmap = 'seismic')
#     ax = plt.gca()
#     plt.savefig("y_fluc_norm_first_"+k+".png")
#     plt.close()
#
# for k in inputsfirst.keys():
#     plt.figure()
#     plt.imshow(inputsfirst[k][2])
#     plt.savefig("y_smooth_" + k + ".png")
#     plt.close()
"""
Ns = np.loadtxt("ICM_old/data/Ns.txt")
amp_pressure = np.zeros((1500,6))
Functions.plot_pressure_spectrum(ells_uncoupled, cl00_uncoupled, Ns, amp_pressure, 27.052, std_amp)

# Look at them beautiful plots
plt.figure()
plt.loglog(Functions.radial_profile(y_smooth_gaussian,[70,70]))
radialmean, radialstd = Functions.radial_profilestat(y_smooth_gaussian,[70,70])
print(np.shape(radialmean[0]))
plt.loglog(radialmean[0])
plt.fill_between((radialmean[1][:-1]+radialmean[1][1:])/2, radialmean[0]+radialstd[0]/2,radialmean[0]-radialstd[0]/2)
plt.show()

Functions.plot_y_fluc(y_fluc_ellipse)
Functions.plot_y_fluc(y_fluc_gaussian)
Functions.plot_y_fluc(y_fluc_wavelet)
Functions.plot_y_fluc(y_fluc_norm_ellipse)
Functions.plot_y_fluc(y_fluc_norm_gaussian)
Functions.plot_y_fluc(y_fluc_norm_wavelet)
plt.imshow(y_fluc_norm_wavelet,vmin=np.min(y_fluc_norm_gaussian),vmax=np.max(y_fluc_norm_gaussian))
plt.plot(y_smooth_ellipse[0,70:])
plt.imshow(y_smooth_gaussian)

#plt.imshow(y_smooth_wavelet)
temp = (y_smooth_wavelet - y_smooth_gaussian)
plt.imshow(temp)
#plt.hist(temp.flatten(),bins=64)
print(np.min(temp))
print(np.max(temp))
"""

# In[21]:


