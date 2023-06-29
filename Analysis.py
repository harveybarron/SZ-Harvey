from astropy import fits
from Functions import *
hudl = fits.open('ICM_old/data/map2048_MILCA_Coma_20deg_G.fits')
data = hudl[1].data
norm_y_fluc = np.loadtxt('ICM_old/data/normalised_y_fluc.txt')
norm_y_fluc_gaussianblur = np.loadtxt('normalised_y_fluc_gaussianblur.txt')

outputk, output_spec, other_shit = namaster_spectrum(norm_y_fluc,6,500,2000,27.052,1.7177432059, 15,60)
plot_this_shit

outputk, output_spec, other_shit = namaster_spectrum(norm_y_fluc_gaussianblur,6,500,2000,27.052,1.7177432059, 15,60)

#LOAD Chandra data
outputk, output_spec, other_shit = namaster_spectrum(norm_y_fluc_chandra,bin_size_chandra,500,2000,27.052,1.7177432059, 15,60)

#LOAD VLA data
outputk, output_spec, other_shit = namaster_spectrum(norm_y_fluc_VLA,6,500,2000,27.052,1.7177432059, 15,60)