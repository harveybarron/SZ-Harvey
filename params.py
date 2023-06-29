from astropy.io import fits
import numpy as np
hudl = fits.open('ICM_old/data/map2048_MILCA_Coma_20deg_G.fits')
hudl1 = fits.open('ICM_old/data/map2048_MILCA_Coma_20deg_first.fits')
hudl2 = fits.open('ICM_old/data/map2048_MILCA_Coma_20deg_last.fits')
hudlnfw = fits.open('Coma_ymap.fits')
data = hudl[1].data
data1 = hudl1[1].data
data2 = hudl2[1].data
datanfw = hudlnfw[1].data
data = (data1 + data2)/2
#[279:419,282:422]
norm_y_fluc = np.loadtxt('ICM_old/data/normalised_y_fluc.txt')
norm_y_fluc_gaussianblur = np.loadtxt('data/normalised_y_fluc_gaussianblur.txt')
#Elliptical model fitting parameters
emp={'e1':[[352.42445296,349.85768166,1,0], 1.7177432059, 700, 700, 240]}
gaussian = {'g2':2,'g4':4,'g6':6,'g8':8,'g10':10,'g5':5}
wavelet = {'w_db3_4_0.15':['db3',4,0.15],'w_db3_4_0.20':['db3',4,0.20],'w_db3_4_0.25':['db3',4,0.25],'w_db3_4_0.30':['db3',4,0.30],'w_db3_4_0.35':['db3',4,0.35]}
nfw = {'n1':1}
# Namaster parameters
namp=(6, 500, 2000, 27.052, 1.7177432059, 15, 60)
# Conversion factor to convert to kpc
ell_kpc=(27.052*60*180)

# Khatri's data
khatrix, khatriy, khatrierror = (0.000461089,0.00056334,0.000717526,0.000974452,0.001333839,0.001846432),\
                                (2.89673913,3.336956522,3.798913043,4.027173913,3.586956522,2.934782609),\
                                [[0.380434783,0.434782609,0.505434783,0.586956522,0.679347826,0.657608696],
                                 [0.380434783,0.440217391,0.505434783,0.592391304,0.684782609,0.663043478]]