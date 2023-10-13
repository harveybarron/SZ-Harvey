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
nfw = {'n1':1}
gaussian = {'g2':2,'g4':4,'g6':6,'g8':8,'g10':10,'g15':15,'g20':20}

wavelet = {'w_db3_4_0.10':['db3',4,0.10],'w_db3_4_0.20':['db3',4,0.20],'w_db3_4_0.30':['db3',4,0.30],'w_db3_4_0.40':['db3',4,0.40],'w_db3_4_0.50':['db3',4,0.50],'w_db3_4_0.60':['db3',4,0.60],'w_db3_4_0.70':['db3',4,0.70],'w_db3_4_0.80':['db3',4,0.80],'w_db3_4_0.90':['db3',4,0.90]}
waveletdepth = {'w_db3_1_0.40':['db3',1,0.40],'w_db3_2_0.40':['db3',2,0.40],'w_db3_3_0.40':['db3',3,0.40],'w_db3_4_0.40':['db3',4,0.40]}
wavelettype = {'w_db3_4_0.40':['db3',4,0.40],'w_haar_4_0.40':['haar',4,0.40],'w_dmey_4_0.40':['dmey',4,0.40],'w_sym2_4_0.40':['sym2',4,0.40],'w_coif1_4_0.40':['coif1',4,0.40]}

packet = {'packet_db3_4_0.10':['db3',4,0.10],'packet_db3_4_0.20':['db3',4,0.20],'packet_db3_4_0.30':['db3',4,0.30],'packet_db3_4_0.40':['db3',4,0.40],'packet_db3_4_0.50':['db3',4,0.50],'packet_db3_4_0.60':['db3',4,0.60],'packet_db3_4_0.70':['db3',4,0.70],'packet_db3_4_0.80':['db3',4,0.80],'packet_db3_4_0.90':['db3',4,0.90]}
packetdepth = {'packet_db3_1_0.40':['db3',1,0.40],'packet_db3_2_0.40':['db3',2,0.4],'packet_db3_3_0.40':['db3',3,0.4],'packet_db3_4_0.4':['db3',4,0.4]}
packettype = {'p_db3_4_0.40':['db3',4,0.40],'p_haar_4_0.40':['haar',4,0.40],'p_dmey_4_0.40':['dmey',4,0.40],'p_sym2_4_0.40':['sym2',4,0.40],'p_coif1_4_0.40':['coif1',4,0.40]}

undec = {'undec_bsplineatrous_7_1_0':['BsplineWaveletTransformATrousAlgorithm',7,1,0],'undec_bsplineatrous_7_2_0':['BsplineWaveletTransformATrousAlgorithm',7,2,0],'undec_bsplineatrous_7_3_0':['BsplineWaveletTransformATrousAlgorithm',7,3,0],'undec_bsplineatrous_7_4_0':['BsplineWaveletTransformATrousAlgorithm',7,4,0],'undec_bsplineatrous_7_5_0':['BsplineWaveletTransformATrousAlgorithm',7,5,0],'undec_bsplineatrous_7_6_0':['BsplineWaveletTransformATrousAlgorithm',7,6,0]}
undectype = {'u_bsplineatrous_7_3_0':['BsplineWaveletTransformATrousAlgorithm',7,3,0],'u_linearatrous_7_3_0':['LinearWaveletTransformATrousAlgorithm',7,3,0],'u_haaratrous_7_3_0':['UndecimatedHaarTransformATrousAlgorithm',7,3,0]}
#undecnoise = {'undec_bsplineatrous_7_4_0':['BsplineWaveletTransformATrousAlgorithm',7,4,0],'undec_bsplineatrous_7_4_1':['BsplineWaveletTransformATrousAlgorithm',7,4,1],'undec_bsplineatrous_7_4_2':['BsplineWaveletTransformATrousAlgorithm',7,4,2]}
undecradial = {'undecradial_bsplineatrous_7_1_0':['BsplineWaveletTransformATrousAlgorithm',7,1,0],'undecradial_bsplineatrous_7_2_0':['BsplineWaveletTransformATrousAlgorithm',7,2,0],'undecradial_bsplineatrous_7_3_0':['BsplineWaveletTransformATrousAlgorithm',7,3,0],'undecradial_bsplineatrous_7_4_0':['BsplineWaveletTransformATrousAlgorithm',7,4,0],'undecradial_bsplineatrous_7_5_0':['BsplineWaveletTransformATrousAlgorithm',7,5,0],'undecradial_bsplineatrous_7_6_0':['BsplineWaveletTransformATrousAlgorithm',7,6,0],'undecradial_bsplineatrous_7_7_0':['BsplineWaveletTransformATrousAlgorithm',7,7,0]}
undecradialtype = {'u_bsplineatrous_7_4_0':['BsplineWaveletTransformATrousAlgorithm',7,4,0],'u_linearatrous_7_4_0':['LinearWaveletTransformATrousAlgorithm',7,4,0],'u_haaratrous_7_4_0':['UndecimatedHaarTransformATrousAlgorithm',7,4,0]}

groups = (emp,nfw,gaussian,wavelet,waveletdepth,wavelettype,packet,packetdepth,packettype,undec, undectype, undecradial, undecradialtype)
# Namaster parameters
namp=(6, 500, 2000, 27.2243172641, 1.7177432059087032, 15, 60)
#28.26237629442678 where did this come from!!
# Conversion factor to convert to kpc
ell_kpc=(27.052*60*180)

# Khatri's data
khatrix, khatriy, khatrierror = (0.000461089,0.00056334,0.000717526,0.000974452,0.001333839,0.001846432),\
                                (2.89673913,3.336956522,3.798913043,4.027173913,3.586956522,2.934782609),\
                                [[0.380434783,0.434782609,0.505434783,0.586956522,0.679347826,0.657608696],
                                 [0.380434783,0.440217391,0.505434783,0.592391304,0.684782609,0.663043478]]