import os, sys
import numpy as np
from matplotlib import pyplot
from scipy.integrate import quad, cumtrapz, dblquad, trapz
from scipy.optimize import minimize_scalar
from astropy.io import fits as pyfits
from scipy.special import gamma as Gammafun

def yintegral(zz, theta, theta_s, c_GNFW, a_GNFW, b_GNFW):
  r=np.sqrt(theta**2+zz**2)
  r/=theta_s
  yintegral=(r**(-c_GNFW))*( (1. + (r**(a_GNFW)))**((c_GNFW - b_GNFW)/a_GNFW))
  return yintegral

def yintegral3d(rr, theta_s, c_GNFW, a_GNFW, b_GNFW):
  r=rr/theta_s
  yintegral3d=y_coeff*(r**(-c_GNFW))*( (1. + (r**(a_GNFW)))**((c_GNFW - b_GNFW)/a_GNFW))*4*np.pi*r**2*theta_s**2
  return yintegral3d

def GNFW_YsphVolIntegrand(r, c, a, b):
  return r**(2-c)*(1+r**a)**((c-b)/a)

def y_int_func(y, x, theta_s, c, a, b, thetalimit):
  r=np.sqrt(x**2+y**2)
  if r==0.:
    return 2.0*theta_s*Gammafun((1.0-c)/a)*Gammafun((b-1.0)/a)/Gammafun((-c+b)/a)/a
  else:
    return quad(yintegral, -thetalimit, thetalimit, args=(r, theta_s, c, a, b), epsrel=1e-4)[0]

def pixel_integrand(y, x, r, func):
  r1=np.sqrt(x**2+y**2)
  return np.interp(r1, r, func)

def Yfrac_diff(r, c, a, b, Y_int, Yfrac_targ):
    test=quad(GNFW_YsphVolIntegrand, 0., r, epsrel=1e-4, args=(c, a, b))[0]/Y_int
    return np.abs(test-Yfrac_targ)

def make_maps(c, a, b, c500, theta_s, Ytot, clus_name, do_plots=False):
  theta500=theta_s*c500
  #thetalimit=5*theta500
  #print 'thetalimit = 5*theta500 = ', thetalimit
  # Set integration limits such that 95% of Ytot is contained within the integration radius (spherical integrals) OR if too large, 10*theta_s
  Yfrac=0.95
  Y_int=Gammafun((3-c)/a)*Gammafun((b-3)/a)/Gammafun((-c+b)/a)/a
  test=quad(GNFW_YsphVolIntegrand, 0., 10., epsrel=1e-4, args=(c, a, b))[0]/Y_int
  if test<Yfrac:
      thetalimit = 10*theta_s
  else:
      thetalimit = minimize_scalar(Yfrac_diff, args=(c, a, b, Y_int, Yfrac)).x
      thetalimit*=theta_s
  print('thetalimit = '+str(thetalimit)+', 95% of Ytot contained within this spherical radius')
  
  # Calculate the spherical integral of the electron pressure
  Y_int=Gammafun((3-c)/a)*Gammafun((b-3)/a)/Gammafun((-c+b)/a)/a
  Y500_int=quad(GNFW_YsphVolIntegrand, 0., c500, epsrel=1e-4, args=(c, a, b))[0]
  print('Y500 = Ytot/ '+str(Y_int/Y500_int))
  Y5R500_int=quad(GNFW_YsphVolIntegrand, 0., 5*c500, epsrel=1e-4, args=(c,a,b))[0]
  print('Y5R500 = Ytot/ '+str(Y_int/Y5R500_int))
  
  #  Calculate the integral of the electron pressure over line of sight at the centre
  y0_int = Gammafun((1.0-c)/a)*Gammafun((b-1.0)/a)/Gammafun((-c+b)/a)/a
  #  Use these to get the normalisation constants for y
  y_coeff = Ytot/Y_int/(theta_s**3)/4.0/np.pi
  #  y value at the centre
  y0 = 2.0*y_coeff*theta_s*y0_int

  print('y value at the centre is '+str(y0))

  #  Calculate y(r) for comparison
  halfsize=100
  thetamin=1e-3
  #  Make the limit a little bit bigger to avoid interpolation problems later
  logthetalim1 = np.log10(thetalimit*1.1)
  theta=np.zeros((halfsize))
  theta[1:]=np.logspace(np.log10(thetamin), logthetalim1, halfsize-1)

  #  Project pressure and integrate over line of sight to calculate y(r)
  yarray=np.zeros((halfsize))
  for i, thetai in enumerate(theta):
    #rlimit1=np.sqrt( max(thetalimit*thetalimit - thetai*thetai, 0) )
    if thetai==0:
      yarray[i]=y0
    #elif rlimit1>0:
    else:
      yarray[i]=quad(yintegral, 0, np.inf, args=(thetai, theta_s, c, a, b), epsrel=1e-4)[0]*2*y_coeff
   
  # Now construct pixel maps of the quantities, averaging over pixels
  # Can get pixel size etc from a template FITS file or set them manually
  f=pyfits.open('ICM_old/data/map2048_MILCA_Coma_20deg_G.fits')
  res=np.abs(f[1].header['cdelt1'])*60
  npix=int(f[1].header['naxis1'])
  crpix=int(npix/2)
  f[1].header['crpix1']=crpix
  f[1].header['crpix2']=crpix

  ymap=np.zeros((npix,npix))
  x=(np.arange(npix)-crpix)*res
  X, Y=np.meshgrid(x, x)
  R=np.sqrt(X**2+Y**2)
  # Firstly, calculate a pixel-averaged ymap value as a function of r, finding where it converges to y(r) for this resolution
  r=np.arange(0., np.max(R), res/4)
  yarray2=np.interp(r, theta, yarray, right=0.)
  yarray_av=np.zeros_like(yarray2)
  yarray_av2=np.zeros_like(yarray2)
  for ipix, thetai in enumerate(r):
    if (thetai+res/2)>np.max(theta):
      rbreak=thetai
      break
    yarray_av[ipix]=dblquad(y_int_func, thetai-res/2, thetai+res/2, lambda x: -res/2, lambda x: res/2., args=(theta_s,c,a,b,thetalimit), epsrel=1e-3)[0]*y_coeff/res**2
    yarray_av2[ipix]=dblquad(pixel_integrand, thetai-res/2, thetai+res/2, lambda x: -res/2, lambda x: res/2., args=(theta, yarray), epsrel=1e-3)[0]/res**2
    if np.abs(yarray2[ipix]-yarray_av[ipix])/yarray_av[ipix]<0.01:
      rbreak=thetai
      break

  ipix1=np.nonzero((r==rbreak))[0][0]
  if do_plots:
      pyplot.plot(r, yarray2, '.')
      pyplot.plot(r[:ipix1+1], yarray_av[:ipix1+1], '.r')
      pyplot.xlabel('r / arcmin')
      pyplot.ylabel(r'$y$ / arcmin$^2$')
      pyplot.show()

  yarray_av[ipix1+1:]=yarray2[ipix1+1:]

  # Now, fill in the y-map.  If the radius at the centre of the pixel is less than the radius found above, use the integral, otherwise use y(r)
  for ipix in range(crpix,npix):
    for jpix in range(crpix,npix):
      if R[ipix,jpix]>thetalimit: continue
      if R[ipix,jpix]<rbreak:
        print(ipix, jpix, 'Integrating y')
        ymap[ipix,jpix]=dblquad(y_int_func, Y[ipix,jpix]-res/2, Y[ipix,jpix]+res/2, lambda x: X[ipix,jpix]-res/2, lambda x: X[ipix,jpix]+res/2., args=(theta_s,c,a,b,thetalimit), epsrel=1e-3)[0]*y_coeff/res**2
      else:
        ymap[ipix,jpix]=np.interp(R[ipix,jpix], theta, yarray)

  # Other 3 quadrants are symmetrical
  ymap[1:crpix,crpix:]=ymap[crpix+1:,crpix:][::-1,:]
  ymap[crpix:,1:crpix]=ymap[crpix:,crpix+1:][:,::-1]
  ymap[1:crpix,1:crpix]=ymap[crpix+1:,crpix+1:][::-1,::-1]

  # Check: np.sum(ymap)*res**2 ~ Ytot
  print('sum(ymap)*pixsize**2 = '+str(np.sum(ymap)*res**2))
  print('Should be roughly equal to Ytot = '+str(Ytot))

  # Write out the y-map
  f[1].data=ymap
  f[1].header['BUNIT']='COMPTON-Y'
  f.verify('fix')
  f.writeto(clus_name+'_ymap.fits', overwrite=True)
  f.close()

  return 

if __name__=='__main__':
  # Arnaud universal GNFW parameters
  #a=1.0510
  #b=5.4907
  #c=0.3081
  #c500=1.177

  # Cluster parameters
  # Coma - from Yvette's analysis
  theta_s=46.5 # arcmin
  Ytot=0.156 # arcmin^2
  a=1.41
  b=6.11
  c=0.43 # fixed to XCOP value
  c500=1.03
  
  do_plots=False

  clus_name='Coma'

  make_maps(c, a, b, c500, theta_s, Ytot, clus_name, do_plots)

  sys.exit(0)
