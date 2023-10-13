import AnalysisEngine
import pickle
import numpy as np
from astropy.io import fits
import Functions
#import matplotlib as mpl
#mpl.use('Qt5Agg')
import matplotlib.pyplot as plt
import skimage.io
import skimage.filters
import pymaster as nmt
import pysap

from params import *

def beam(l,source='Planck'):
    """params:
        -> l: array-like; ell values
        -> source: string
       returns:
        -> array-like; value of beam at those ells
    """
    if source == 'Planck':
        res=10./60
        sig=res/2.3548
    return np.exp(-0.5*l*(l+1)*(sig*np.pi/180)**2)

def namaster_spectrum_khatri(norm_y_fluc, bin_number, minscale, maxscale, arcmin2kpc, pixsize, theta_ap, cr, norm_y_fluc1=0):
    import pymaster as nmt
    if norm_y_fluc1.all() == 0:
        norm_y_fluc1 = norm_y_fluc

    Lx = 4. * np.pi / 180
    Ly = 4. * np.pi / 180
    Nx, Ny = len(norm_y_fluc), len(norm_y_fluc)
    mask = np.zeros((Nx, Ny))
    # the centre will always be the middle elements becuase of the way the fluctuation maps have been computed
    cen_x, cen_y = Nx / 2., Ny / 2.
    # cr = 60  # radius of mask in arcmin
    I, J = np.meshgrid(np.arange(mask.shape[0]), np.arange(mask.shape[1]))
    dist = np.sqrt((I - cen_x) ** 2 + (J - cen_y) ** 2)
    dist = dist * pixsize
    idx = np.where(dist <= cr)
    # theta_ap = 15  # apodization scale in arcmin
    mask[idx] = 1 - np.exp(-9 * (dist[idx] - cr) ** 2 / (2 * theta_ap ** 2))  # described in Khatri et al.

    #Creating bins
    # l's have to be converted from kpc using l = pi/angular sep
    # We want to use bin sizes between 500 and 2000 kpc in terms of l's
    # l_min = (180 * 60 * arcmin2kpc / maxscale)
    # l_max = (180 * 60 * arcmin2kpc / minscale)
    # l_min = 120.52290442290746
    # l_max = 620.4107753722085
    #
    # bin_size = (l_max - l_min) / bin_number
    #
    # l0_bins = []
    # lf_bins = []
    #
    # for i in range(bin_number):
    #     l0_bins.append(l_min + bin_size * i)
    #     lf_bins.append(l_min + bin_size * (i + 1))


    #effective l's
    # b = nmt.NmtBinFlat(l0_bins, lf_bins)
    # ells_uncoupled = b.get_effective_ells()
    # lambdas_inv = ells_uncoupled / (arcmin2kpc * 60 * 180)
    # k = 2 * np.pi * lambdas_inv

    ells_uncoupled_khatri = tuple([x * arcmin2kpc * 60 * 180 for x in khatrix])
    ells_uncoupled_khatri_mid = [10**((np.log10(s)+np.log10(t))/2) for s, t in zip(ells_uncoupled_khatri, ells_uncoupled_khatri[1:])]
    print(ells_uncoupled_khatri_mid)
    ells_uncoupled_khatri_diff = [t-s for s, t in zip(ells_uncoupled_khatri, ells_uncoupled_khatri[1:])]
    print(min(ells_uncoupled_khatri_diff))
    khatridiammax = min(ells_uncoupled_khatri_diff)/2

    l0_bins_klow = []
    lf_bins_klow = []


    for i in range(6):
        l0_bins_klow.append(ells_uncoupled_khatri[i]-khatridiammax)
        lf_bins_klow.append(ells_uncoupled_khatri[i]+khatridiammax)

    # khatrififthlog = (np.log10(ells_uncoupled_khatri[5]) - np.log10(ells_uncoupled_khatri[0]))/5
    # l0_bins_klow.append(10**(np.log10(ells_uncoupled_khatri[0]) - khatrififthlog/2))
    # for i in range(5):
    #     lf_bins_klow.append(10**(np.log10(l0_bins_klow[i]) + khatrififthlog))
    #     l0_bins_klow.append(lf_bins_klow[i])
    # lf_bins_klow.append(10 ** (np.log10(l0_bins_klow[5]) + khatrififthlog))

    # khatrififth = (ells_uncoupled_khatri[5]-ells_uncoupled_khatri[0])/5
    # l0_bins_klow.append(ells_uncoupled_khatri[0] - khatrififth/2)
    # lf_bins_klow.append(ells_uncoupled_khatri[0] + khatrififth / 2)
    # for i in range(5):
    #     l0_bins_klow.append(l0_bins_klow[i] + khatrififth)
    #     lf_bins_klow.append(lf_bins_klow[i] + khatrififth)


    # for i in range(5):
    #     if i == 0:
    #         l0_bins_klow.append(ells_uncoupled_khatri[i] - ells_uncoupled_khatri_mid[i]/2)
    #     else:
    #         l0_bins_klow.append(lf_bins_klow[i-1])
    #     lf_bins_klow.append(ells_uncoupled_khatri[i] + ells_uncoupled_khatri_mid[i]/2)
    # l0_bins_klow.append(lf_bins_klow[4])
    # lf_bins_klow.append(ells_uncoupled_khatri[5] + ells_uncoupled_khatri_mid[4] / 2)

    # l0_bins_klow.append(2 * ells_uncoupled_khatri[0] - ells_uncoupled_khatri_mid[0])
    # for i in range(5):
    #     l0_bins_klow.append(ells_uncoupled_khatri_mid[i])
    #     lf_bins_klow.append(ells_uncoupled_khatri_mid[i])
    # lf_bins_klow.append(2 * ells_uncoupled_khatri[5] - ells_uncoupled_khatri_mid[4])

    print("l0bins k")
    print(l0_bins_klow)
    print("lfbins k")
    print(lf_bins_klow)
    bklow = nmt.NmtBinFlat(l0_bins_klow, lf_bins_klow)
    ells_uncoupled_bklow = bklow.get_effective_ells()
    print(ells_uncoupled_bklow)
    print(ells_uncoupled_khatri)
    f0 = nmt.NmtFieldFlat(Lx, Ly, mask, [norm_y_fluc], beam=[ells_uncoupled_bklow, beam(ells_uncoupled_bklow)])
    f1 = nmt.NmtFieldFlat(Lx, Ly, mask, [norm_y_fluc1], beam=[ells_uncoupled_bklow, beam(ells_uncoupled_bklow)])

    ## ANGULAR POWER SPECTRUM
    w00 = nmt.NmtWorkspaceFlat()
    w00.compute_coupling_matrix(f0, f1, bklow)

    #Coupling matrix used to estimate angular spectrum
    cl00_coupled = nmt.compute_coupled_cell_flat(f0, f1, bklow)
    cl00_uncoupled = w00.decouple_cell(cl00_coupled)[0]
    amp = abs((ells_uncoupled_bklow**2)*cl00_uncoupled/(2*np.pi))**(1/2)
    print("amp")
    print(amp)
    ## Covariance matrix
    cw = nmt.NmtCovarianceWorkspaceFlat()
    cw.compute_coupling_coefficients(f0, f1, bklow)
    covar = nmt.gaussian_covariance_flat(cw, 0, 0, 0, 0, ells_uncoupled_bklow,
                                         [cl00_uncoupled], [cl00_uncoupled],
                                         [cl00_uncoupled], [cl00_uncoupled], w00)
    std_power = (np.diag(covar))
    std_amp = np.sqrt(abs((ells_uncoupled_bklow**2)*std_power/(2*np.pi))**(1/2))

    return amp, std_amp, cl00_uncoupled, covar, ells_uncoupled_bklow

def aravalo_mark(norm_y_fluc, bin_number, minscale, maxscale, arcmin2kpc, pixsize, theta_ap, cr, norm_y_fluc1=0):
    from AnalysisEngine.analysis_engine.statistics.spectra import per_spectra, arevalo_spectra, flatsky_spectra, spectra_base
    if norm_y_fluc1.all() == 0:
        norm_y_fluc1 = norm_y_fluc

    Lx = 4. * np.pi / 180
    Ly = 4. * np.pi / 180
    Nx, Ny = len(norm_y_fluc), len(norm_y_fluc)
    mask = np.zeros((Nx, Ny))
    cen_x, cen_y = Nx / 2., Ny / 2.
    # cr = 60  # radius of mask in arcmin
    I, J = np.meshgrid(np.arange(mask.shape[0]), np.arange(mask.shape[1]))
    dist = np.sqrt((I - cen_x) ** 2 + (J - cen_y) ** 2)
    dist = dist * pixsize
    idx = np.where(dist <= cr)
    # theta_ap = 15  # apodization scale in arcmin
    mask[idx] = 1 - np.exp(-9 * (dist[idx] - cr) ** 2 / (2 * theta_ap ** 2))  # described in Khatri et al.
    #mask = nmt.mask_apodization_flat(mask, Lx, Ly, aposize=0.25, apotype="C1")
    #mask = np.ones_like(norm_y_fluc)

    def l_to_k(x):
        return (2. * np.pi) / x

    def get_l(fitsfile, redshift, Nx, units='kpc'):
        from astropy.cosmology import WMAP1 as cosmo
        import astropy.units as u
        from astropy.io import fits
        d_A = cosmo.angular_diameter_distance(z=redshift)
        fsz = fits.open(fitsfile)
        dtheta = (fsz[1].header['CDELT2'] * u.deg)
        dx = (dtheta * d_A).to(u.kpc, u.dimensionless_angles())
        Ltheta = dtheta * Nx
        if units == 'kpc':
            L = dx * Nx
        elif units == 'degrees':
            L = Ltheta
        elif units == 'radians':
            L = Ltheta.to(u.rad)
        return L


    Lradians = get_l('ICM_old/data/map2048_MILCA_Coma_20deg_G.fits',0.023,140,'radians').value
    L = get_l('ICM_old/data/map2048_MILCA_Coma_20deg_G.fits',0.023,140,'kpc').value

    NBINS = 50
    minb, maxb = 1., None
    #minb, maxb =  L/(2500), L/(500)

    k1, fek1 = flatsky_spectra.modal_spectrum(norm_y_fluc, ar2=norm_y_fluc1, mask=mask, lenn=(L, L), min_bin=minb, max_bin=maxb, log_space=False, nbins=NBINS)

    lags = arevalo_spectra.k_to_discrete_lags(k1, Nx, L=L)
    k, fek = arevalo_spectra.modal_spectrum(norm_y_fluc*mask, ar2=norm_y_fluc1*mask, exp1=mask, exp2=mask, lenn=L, lags=lags)

    k2_3, fek2_3 = per_spectra.modal_spectrum(norm_y_fluc*mask, ar2=norm_y_fluc1*mask, lenn=(L, L))
    k2, fek2, _ = spectra_base.spectrum_integrate(k2_3, fek2_3, spec_type='modal', lenn=(L, L), bin_center=True)

    amp = abs((k ** 2) * fek / (2 * np.pi)) ** (1 / 2)
    amp1 = abs((k1 ** 2) * fek1 / (2 * np.pi)) ** (1 / 2)
    amp2 = abs((k2**2) * fek2 / (2. * np.pi))**(1./2.)

    return amp,k,fek,amp1,k1,fek1,amp2,k2,fek2

inputsfirst = pickle.load(open('inputsfirst.p', 'rb'))
inputslast = pickle.load(open('inputslast.p', 'rb'))
outputs = {}


for k in inputsfirst.keys():
    print(k)
    #outputs[k] = namaster_spectrum_khatri(inputsfirst[k][0], namp[0],
                 #namp[1],namp[2],namp[3],namp[4],namp[5],namp[6],norm_y_fluc1=inputslast[k][0])
    outputs[k] = aravalo_mark(inputsfirst[k][0], namp[0],namp[1], namp[2], namp[3], namp[4], namp[5], namp[6],norm_y_fluc1=inputslast[k][0])

#Plot the spectrum
# plt.figure()
# plt.xscale('log')
# cmap = plt.get_cmap('tab10')
# colour = 0
# for k in wavelet.keys():
#         lambdas_inv = outputs[k][4]/ell_kpc
#         amp=outputs[k][0]
#         std_amp=outputs[k][1]
#         # Multiply by 1e6 to compare with Khatri
#         #plt.errorbar(lambdas_inv,amp*1e6, yerr=std_amp*1e6, fmt='b.', ecolor=cmap(colour),elinewidth=1,capsize = 4,label="depth = 1")
#         plt.plot(lambdas_inv,amp*1e6,color=cmap(colour),label="w="+str(wavelet[k][0])+", depth="+str(wavelet[k][1])+", t="+str(wavelet[k][2]))
#         colour += 1
# plt.errorbar(khatrix,khatriy,yerr=(khatrierror), fmt='r.',ecolor='black',elinewidth=1,capsize = 4)
# plt.xlabel("$1/\lambda$ ($kpc^{-1}$)")
# plt.ylabel("Amplitude of power spectrum")
# plt.legend()
# plt.title("Power Spectrum of wavelet")
# plt.savefig('crosspowerspectrum_khatribins_wavelettest.png')

# plt.figure()
# plt.xscale('log')
# plt.yscale('log')
# cmap = plt.get_cmap('tab10')
# colour = 0
# for k in outputs.keys():
#         plt.plot(outputs[k][1],outputs[k][2],color=cmap(colour))
#         colour += 1
# plt.savefig('aravalopowerspectrummarkkhatri.png')


colour = 0
for g in groups:
    plt.figure()
    plt.xscale('log')
    #plt.yscale('log')
    cmap = plt.get_cmap('tab10')
    name = 0
    for k in g.keys():
        if name == 0:
            name = k

        laminv0 = outputs[k][1] / (2*np.pi)
        amp = outputs[k][0]*1e6
        fek = outputs[k][2]

        amp1 = outputs[k][3]*1e6
        laminv1 = outputs[k][4] / (2*np.pi)
        fek1 = outputs[k][5]

        amp2 = outputs[k][6]*1e6
        laminv2 = outputs[k][7] / (2*np.pi)
        fek2 = outputs[k][8]

        # amp3 = outputs[k][9]*1e6
        # laminv3 = outputs[k][10] / (2*np.pi)
        # fek3 = outputs[k][11]
        #
        # amp4 = outputs[k][12]*1e6

        # Multiply by 1e6 to compare with Khatri
        plt.plot(laminv0, np.abs(amp), color=cmap(colour))
        plt.plot(laminv1, np.abs(amp1), color=cmap(colour + 1), linestyle='--')
        #plt.plot(laminv1[amp1 > 0.], amp1[amp1 > 0.], color=cmap(colour+1), linestyle='--')
        plt.plot(laminv2, np.abs(amp2), color=cmap(colour+2), linestyle=':')
        #plt.plot(laminv3, np.abs(amp3), color=cmap(colour+3), linestyle='--')
        #plt.plot(laminv2, np.abs(amp4), color=cmap(colour+2), linestyle=':')

        colour = colour + 1

    plt.errorbar(khatrix,khatriy,yerr=(khatrierror), fmt='r.',ecolor='black',elinewidth=1,capsize = 4)
    plt.xlabel("$1/\lambda$ ($kpc^{-1}$)")
    plt.ylabel("$A_{\delta y} = (k^2P_{\delta y}(k)/(2\pi))^{1/2}(10^{-6})$")
    # plt.legend()
    plt.title("Comparison")
    # plt.show()
    plt.plot([], [], label='Cross Arevalo')
    plt.plot([], [], linestyle='--', label='Cross Flatsky')
    plt.plot([], [], linestyle=':', label='Periodogram')
    plt.legend()
    plt.savefig('NEWmarkpowerspectrum_' + name + '_compareflatbeamed6.pdf')
    plt.cla()





# for k in emp.keys():
#         lambdas_inv = outputs[k][1]/ell_kpc
#         amp=outputs[k][0]
#         lambdas_inv1 = outputs[k][4] / ell_kpc
#         amp1 = outputs[k][3]
#         # Multiply by 1e6 to compare with Khatri
#         plt.plot(lambdas_inv, amp * 1e6, color=cmap(colour))
#         plt.plot(lambdas_inv1, amp1 * 1e6, color=cmap(colour))
# colour += 1
# for k in gaussian.keys():
#     lambdas_inv = outputs[k][1] / ell_kpc
#     amp = outputs[k][0]
#     lambdas_inv1 = outputs[k][4] / ell_kpc
#     amp1 = outputs[k][3]
#     # Multiply by 1e6 to compare with Khatri
#     plt.plot(lambdas_inv, amp * 1e6, color=cmap(colour))
#     plt.plot(lambdas_inv1, amp1 * 1e6, color=cmap(colour))
# colour += 1
# for k in wavelet.keys():
#     lambdas_inv = outputs[k][1] / ell_kpc
#     amp = outputs[k][0]
#     lambdas_inv1 = outputs[k][4] / ell_kpc
#     amp1 = outputs[k][3]
#     # Multiply by 1e6 to compare with Khatri
#     plt.plot(lambdas_inv, amp * 1e6, color=cmap(colour))
#     plt.plot(lambdas_inv1, amp1 * 1e6, color=cmap(colour))
