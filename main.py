import numpy as np
from astropy.io import fits
from codes import Computing_ys_in_annuli

hudl = fits.open('ICM_old/data/map2048_MILCA_Coma_20deg_G.fits')
hudl.info()
Computing_ys_in_annuli.get_2Dys_in_annuli(elliptical_model_vals)