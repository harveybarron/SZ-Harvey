
"""
data_generator.py

Provides methods for generating random fields of specific power law (inspired by [1,2])

Functions
---------
generate_field\n
dephase_data\n
add_noise\n
generate_mask\n
generate_exposure\n

Usage:
------
>>> from analysis_engine.simulator import data_generator
>>> data = data_generator.generate_field((256, 256), 11./8.)

References:
-----------
[1] Koch, Eric W., Erik W. Rosolowsky, Ryan D. Boyden, Blakesley Burkhart, Adam Ginsburg, Jason L. Loeppky, and Stella SR Offner.
    "TurbuStat: Turbulence statistics in python." The Astronomical Journal 158, no. 1 (2019): 1.
[2] https://turbustat.readthedocs.io/en/latest/
"""

import numpy as np

import scipy.fft as fft



def generate_field(dims, a1, a2=None, kb=1., A=1., delta=1., lenn=None, seed=1234):
    """generate_field(dims, a1, a2, kb, A, delta, lenn, seed)

    Generates a fBM field of arbitrary dimensions with broken powerlaw
        `a1` and `a2` with amplitude `A` at the break scale `kb`

    https://gist.github.com/cgobat/12595d4e242576d4d84b1b682476317d

    Args:
        dims (tuple): Number of points in each dimension
        a1 (float): The power law
        a2 (float): The power law after the break
        kb (float): The break scale/location
        A (float): The amplitude of the spectrum to return
        delta (float): The smoothing factor
        lenn (tuple): Length of the domain for each dimension
        seed (float): Random number generator seed
    Returns:
        np.ndarray: fBM field of specific powerlaw
    """
    np.random.seed(seed)
    assert len(set(dims)) <= 1, 'Must have the same number of points in each dimension'
    if lenn is None:
        lenn = [2. * np.pi for _ in dims]
    assert len(set(lenn)) <= 1, 'Must be the same length in each dimension'

    if a2 is None:
        a2 = a1

    dx = [lenn[i]/dims[i] for i in range(len(dims))]
    dk = [1./lenn[i] for i in range(len(dims))]

    # Generate wavenumbers
    kvec = [np.fft.fftfreq(dims[i])*dims[i] for i in range(len(dims))]
    kgrid = np.meshgrid(*kvec, indexing='ij')
    kk = np.linalg.norm(kgrid, axis=0)
    kk[kk == 0] = np.nan

    # Random phases
    phases = np.random.uniform(0, 2.*np.pi, size=dims)
    phases = np.cos(phases) + 1j * np.sin(phases)

    # Generate fBM in Fourier space
    a1 *= -1.
    a2 *= -1.
    output = np.sqrt(A*(kk/kb)**(-a1) * (0.5 * (1. + (kk/kb)**(1./delta)))**((a1 - a2)*delta))
    output = output.astype('complex') * phases / np.sqrt(np.prod(dx))
    output[np.isnan(output)] = 0. + 1j * 0.0

    # DC components must have no imaginary components
    index = [0 for _ in dims]
    output[tuple(index)] = output[tuple(index)].real + 1j * 0.0
    index[-1] = -1
    output[tuple(index)] = output[tuple(index)].real + 1j * 0.0

    # Convert to configuration-space
    fbm_field = np.prod(dk) * fft.ifftn(output, norm='forward').real
    return fbm_field

def dephase_data(ar, seed=1234):
    """dephase_data(ar)

    Retains the same power law as the data, but randomizes the phases.
    This creates an fBM field from some raw data.

    Args:
        ar (np.ndarray): Data to dephase
    Returns:
        np.ndarray: Dephased data    
    """
    np.random.seed(seed)
    far = fft.fftn(ar)
    phases = np.random.uniform(0, 2.*np.pi, size=ar.shape)
    far_dephased = np.abs(far) * (np.cos(phases) + 1j * np.sin(phases))
    fbm_field = fft.ifftn(far_dephased).real
    return fbm_field

def add_noise(ar, peak=1.):
    """add_noise(ar, peak)

    Add Poisson noise to an image

    Args:
        ar (np.ndarray): Distribution of Poisson mean values
        peak (float): Modification value for the mean
    Returns:
        np.ndarray: `ar` with noise
    """
    return np.random.poisson(ar * peak) / peak

def generate_mask(shape, p=0.5, seed=1234):
    """generate_mask(shape)

    Creates a mask that would remove 'bad' data. 

    Args:
        shape (tuple): Shape of the mask
        p (float): Fraction of data to remove
    Returns:
        np.ndarray: Mask to indicate bad data
    """
    assert len(set(shape)) <= 1, 'Must have the same number of points in each dimension'
    np.random.seed(seed)
    mask = np.ones(shape)

    n = len(shape)
    grid = np.indices(shape)

    def create_circle(pos, r):
        rvecs = [grid[i,...] - pos[i] for i in range(n)]
        c = np.linalg.norm(rvecs, axis=0)
        return c <= r

    total = np.prod(shape)
    while np.sum(mask)/total > p:
        pos = [np.random.randint(0, shape[i]) for i in range(n)]
        r = np.random.randint(1, shape[0]//16)
        m = create_circle(pos, r)
        mask[m] = 0.

    return mask

def generate_exposure(shape, seed=1234):
    """generate_exposure(shape, seed)

    Args:
        shape (tuple): Shape of the exposure map
    Returns:
        np.ndarray: Exposure map
    """
    assert len(set(shape)) <= 1, 'Must have the same number of points in each dimension'
    np.random.seed(seed)
    exp = np.ones(shape)

    n = len(shape)
    grid = np.indices(shape)

    def create_circle(pos, r):
        rvecs = [grid[i,...] - pos[i] for i in range(n)]
        c = np.linalg.norm(rvecs, axis=0)
        return c <= r

    total = np.prod(shape)
    while np.sum(exp) < 10.*total:
        pos = [np.random.randint(0, shape[i]) for i in range(n)]
        r = np.random.randint(1, shape[0]//16)
        m = create_circle(pos, r)
        exp[m] = exp[m] + 1.

    exp = (exp - np.min(exp))/np.ptp(exp)
    return exp

from astropy.io import fits
import pickle
x2 = generate_field((700,700), a1=-11./3.)
hudlnfw = fits.open('Coma_ymap.fits')
datanfw = hudlnfw[1].data
inputsfirst = pickle.load(open('inputsfirst.p', 'rb'))
g5pad = np.pad(inputsfirst['g5'][2], 280)
ellipsemax = np.max(np.abs(inputsfirst['e1'][0]))
#[279:419,282:422]
testg5 = g5pad + datanfw
x2 = x2*ellipsemax/np.max(np.abs(x2))
testsim = datanfw + x2
simg5 = g5pad +x2
print(np.min(x2))
print(np.max(x2))
print(ellipsemax)
np.savetxt('sim.txt',x2)
import matplotlib.pyplot as plt
plt.imshow(simg5)
plt.savefig("simg5.png")
np.savetxt('simg5.txt',simg5)