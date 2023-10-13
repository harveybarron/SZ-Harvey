import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
import scipy.stats
import skimage.io
import skimage.filters
import pymaster as nmt
from matplotlib.colors import TwoSlopeNorm

def get_2Dys_in_annuli(data, elliptical_model_vals, pixsize):
    """
    params:
        -> data: fit.open()[1].data
        -> elliptical_model_vals: Array-like with 4 elements
        -> elliptical_model_vals[0]: x coordinate of the centre of the Cluster
        -> elliptical_model_vals[1]: y coordinate of the centre of the Cluster
        -> elliptical_model_vals[2]: Range:(0,1]; Skewness of an Elliptical model
        -> elliptical_model_vals[3]: Range:[0,pi]; Rotation of the major axis from the x-axis
        -> pixsize: pixel size, COMA default = 1.7177432059
    returns:
        -> rs: arraylike; The centre values for each bin
        -> ys: arraylike; The average ys in bins
        -> step_size: The size of each bin in arcmins
        -> elliptical_model_vals: Described above
    """

    r_initial = 0
    r_final = 180  # arcmin
    step_size = 3  # arcmin
    cen_x, cen_y, skew, theta = elliptical_model_vals

    X, Y = np.meshgrid(np.arange(data.shape[0]), np.arange(data.shape[1]))

    R_ellipse = np.sqrt((skew*(np.cos(theta))**2 + 1/skew*(np.sin(theta))**2)*(X - cen_x)**2 +
                        (skew*(np.sin(theta))**2 + 1/skew*(np.cos(theta))**2) *
                        (Y - cen_y)**2 + 2*(np.cos(theta))*(np.sin(theta))*(skew-1/skew) *
                        (X - cen_x)*(Y - cen_y))

    rs_boundary = np.arange(r_initial, r_final, step_size)  # computes the values of rs at the boundaries of bins

    ys = []
    rs = []

    for r in (rs_boundary):
        in_ann = np.nonzero((R_ellipse * pixsize > r) & (R_ellipse * pixsize < (r + step_size)))
        av = np.mean(data[in_ann])
        ys.append(av)
        rs.append(r + step_size / 2)

    return np.array(rs), np.array(ys), step_size, elliptical_model_vals, R_ellipse

def gaussian_heatmap(center, image_size, sig = 1):
    """ 
     Produces single gaussian at expected center
    :param center:  the mean position (X, Y) - where high value expected
    :param image_size: The total image size (width, height)
    :param sig: The sigma value
    :return:
    """
    x_axis = np.linspace(0, image_size[0]-1, image_size[0]) - center[0]
    y_axis = np.linspace(0, image_size[1]-1, image_size[1]) - center[1]
    xx, yy = np.meshgrid(x_axis, y_axis)
    kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sig))
    return kernel  

def get_fluctuations_ellipse(data, elliptical_model_vals, pixsize, NAXIS1, NAXIS2, image_length, data_g=0):
    """
    params:
        -> data: fit.open()[1].data
        -> elliptical_model_vals: Array-like with 4 elements,defaultCOMA [349.85768166,352.42445296,1,0]
        -> elliptical_model_vals[0]: x coordinate of the centre of the Cluster
        -> elliptical_model_vals[1]: y coordinate of the centre of the Cluster
        -> elliptical_model_vals[2]: Range:(0,1]; Skewness of an Elliptical model
        -> elliptical_model_vals[3]: Range:[0,pi]; Rotation of the major axis from the x-axis
        -> pixsize: pixel size, COMA default = 1.7177432059
        -> image_length=120*2 #arcmin
    """
    if data_g.all() == 0:
        data_g = data

    rs, ys, step_size, maxval, ellipse = get_2Dys_in_annuli(data_g, elliptical_model_vals, pixsize)
        
     #To account for the region between r=0 and r=(centre of first bin)
    ys_new = np.zeros(len(ys)+1)
    rs_new = np.zeros(len(rs)+1)
    rs_new[1:] = rs
    ys_new[1:] = ys
    ys_new[0]= ys[0]
    x=np.arange(NAXIS1) # where npix is the number of pixels in the x direction, ie NAXIS1 from the header
    y=np.arange(NAXIS2) # as above, NAXIS2=NAXIS1

    x_cen = maxval[0]
    y_cen = maxval[1]
    f = maxval[2]
    theta = maxval[3]

    #Note that centre of the cluster will always be the centre of the image 
    x_ind = np.nonzero(((x-x_cen)*pixsize>=-(image_length/2)) & (((x-x_cen)*pixsize<=(image_length/2))))
    y_ind = np.nonzero(((y-y_cen)*pixsize>=-(image_length/2)) & (((y-y_cen)*pixsize<=(image_length/2))))

    y_fluc = np.zeros((len(x_ind[0]),len(y_ind[0])))
    y_fluc_norm = np.zeros_like(y_fluc)
    y_smooth = np.zeros_like(y_fluc)

    r_ellipse_max = 0.
    for t1,rx in enumerate(x_ind[0]):
        for t2,ry in enumerate(y_ind[0]):
            r_ellipse = np.sqrt((f*(np.cos(theta))**2 + 1/f*(np.sin(theta))**2)*(rx - x_cen)**2  \
                            + (f*(np.sin(theta))**2 + 1/f*(np.cos(theta))**2)*(ry - y_cen)**2  \
                            + 2*(np.cos(theta))*(np.sin(theta))*(f-1/f)*(rx - x_cen)*(ry - y_cen))*pixsize
            if r_ellipse>r_ellipse_max:
                r_ellipse_max = r_ellipse

            y_radius = np.interp(r_ellipse, rs_new, ys_new)
            y_fluc[t1][t2] = data[rx][ry] - y_radius
            y_fluc_norm[t1][t2] = y_fluc[t1][t2]/abs(y_radius)
            y_smooth[t1][t2] = y_radius
            
    return y_fluc, y_fluc_norm, y_smooth
       
def get_fluctuations_gaussian(data, sigma,data_g=0):
    import skimage.io
    import skimage.filters
    if data_g.all() == 0:
        data_g = data

    blurred = skimage.filters.gaussian(data_g, sigma)
    y_fluc = data-blurred
    y_fluc_norm = y_fluc / blurred
    
    return y_fluc, y_fluc_norm, blurred


def get_fluctuations_nfw(data, datanfw, data_g=0):
    if data_g.all() == 0:
        data_g = data

    y_fluc = data - datanfw
    y_fluc_norm = y_fluc / np.abs(datanfw)

    return y_fluc, y_fluc_norm, datanfw

def get_fluctuations_waveletold(data, scale, wavelet='db3',data_g=0,threshold=0.04):
    import pysap
    if data_g.all() == 0:
        data_g = data
    transform_klass = pysap.load_transform(wavelet)
    transform = transform_klass(nb_scale=scale, verbose=1, padding_mode="symmetric")
    transform.data = data_g
    transform.analysis()
    for n in range(0,scale*3+1):
        if n == 0:
            m = np.max(np.max(abs(transform.analysis_data[0]))) * threshold
        else:
            if n % 3 == 1:
                m = np.max((np.max(abs(transform.analysis_data[n])),np.max(abs(transform.analysis_data[n+1])),np.max(abs(transform.analysis_data[n+2]))))*threshold
            if n % 3 == 2:
                m = np.max((np.max(abs(transform.analysis_data[n-1])), np.max(abs(transform.analysis_data[n])), np.max(abs(transform.analysis_data[n + 1])))) * threshold
            if n % 3 == 0:
                m = np.max((np.max(abs(transform.analysis_data[n])), np.max(abs(transform.analysis_data[n - 1])),np.max(abs(transform.analysis_data[n - 2])))) * threshold

        for i in range(0,np.shape(transform.analysis_data[n])[0]):
            for j in range(0,np.shape(transform.analysis_data[n])[1]):
                if abs(transform.analysis_data[n][i][j]) <= m:
                    transform.analysis_data[n][i][j] = 0

    rec_image = transform.synthesis()
    y_fluc = data-rec_image.data
    y_fluc_norm = (data-rec_image.data)/rec_image.data
    return y_fluc, y_fluc_norm, rec_image.data

def get_fluctuations_wavelet(data, level, wavelet='db3',data_g=0,threshold=0.04,thresholdmode='soft',plot=False,plotapprox=False,plotnorm=False,plotlog=False):
    import pywt
    if data_g.all() == 0:
        data_g = data

    if level > pywt.dwtn_max_level(np.shape(data), wavelet):
        print("level greater than maxlevel")
        level = pywt.dwtn_max_level(np.shape(data), wavelet)

    coeffs = pywt.wavedec2(data=data_g, wavelet=wavelet, level=level, mode='symmetric')
    coeffscopy = coeffs.copy()



    for i, value in enumerate(coeffs):
        if i != 0:
            m = np.max(coeffs[i]) * threshold
            print(i,m)
            coeffs[i] = tuple([pywt.threshold(v, m, mode=thresholdmode, substitute=0) for v in coeffs[i]])
            print(np.count_nonzero(coeffs[i])/np.size(coeffs[i]))

    y_smooth = pywt.waverec2(coeffs, wavelet, mode='symmetric')
    y_fluc = data-y_smooth
    y_fluc_norm = y_fluc/y_smooth

    if plot:
        import cmasher as cmr
        colors = cmr.take_cmap_colors('cmr.ocean', len(coeffscopy), cmap_range=(0.15, 0.85), return_fmt='hex')
        plt.figure()
        plt.xscale('log')
        if plotlog:
            plt.yscale('log')
        colour = 0
        for i, value in enumerate(coeffscopy):
            if i != 0 or plotapprox:
                m = np.max(coeffscopy[i]) * threshold
                plt.axhline(y=m, color=colors[colour], linestyle='--')
                flatcoeff = np.array([item for tup in coeffs[i] for item in tup]).flatten()
                flatoldcoeff = np.array([item for tup in coeffscopy[i] for item in tup]).flatten()
                if plotnorm:
                    plt.plot(sorted(np.abs(flatcoeff / np.max(coeffscopy[i])), reverse=True),color=colors[colour], label="scale=" + str(i))
                    plt.plot(sorted(np.abs(flatoldcoeff / np.max(coeffscopy[i])), reverse=True), color=colors[colour],linestyle=':')
                else:
                    plt.plot(sorted(np.abs(flatcoeff), reverse=True), color=colors[colour],label="scale=" + str(i))
                    plt.plot(sorted(np.abs(flatoldcoeff), reverse=True), color=colors[colour], linestyle=':')
            colour += 1
        if plotnorm:
            plt.ylabel("Normalised amplitude of coefficients")
        else:
            plt.ylabel("Amplitude of coefficients")
        plt.legend()
        plt.xlabel("Coefficients sorted by amplitude")
        plt.savefig('waveletcoeff_' + str(wavelet) + '_' + str(level) + str(threshold) + str(thresholdmode) + '.png')
    return y_fluc, y_fluc_norm, y_smooth

def get_fluctuations_waveletpacketold(data, depth, wavelet='db3',data_g=0,threshold=0.04):
    import pywt
    if data_g.all() == 0:
        data_g = data

    wp = pywt.WaveletPacket2D(data=data_g, wavelet=wavelet, mode='symmetric')
    print(wp.maxlevel)
    if depth > wp.maxlevel:
        print("depth greater than maxlevel")
        depth = wp.maxlevel

    new_wp = pywt.WaveletPacket2D(data=np.zeros_like(data), wavelet=wavelet, mode='symmetric',maxlevel=depth)

    for n in range(0,depth+1):
        paths = [node.path for node in wp.get_level(n)]
        leveldata = [node.data for node in wp.get_level(n)]
        #levelthreshold = np.percentile(np.abs(leveldata),100*threshold)
        levelthreshold = np.percentile(np.abs(leveldata), 100)*threshold
        print("levelthreshold: ")
        print(levelthreshold)
        for i,path in enumerate(paths):
            new_wp[path] = pywt.threshold(wp[path].data, levelthreshold, mode='soft', substitute=0)
        leveldatanew = [node.data for node in new_wp.get_level(n)]
        print(100*(1-np.count_nonzero(leveldatanew)/np.size(leveldatanew)))

    y_smooth = new_wp.reconstruct()
    y_fluc = data-y_smooth
    y_fluc_norm = y_fluc/y_smooth
    return y_fluc, y_fluc_norm, y_smooth

def get_fluctuations_waveletpacket(data, depth, wavelet='db3',data_g=0,threshold=0.04,thresholdmode='soft',plot=False,plotapprox=False,plotnorm=False,plotlog=False):
    import pywt
    if data_g.all() == 0:
        data_g = data

    wp = pywt.WaveletPacket2D(data=data_g, wavelet=wavelet, mode='symmetric')
    print(wp.maxlevel)
    if depth > wp.maxlevel:
        print("depth greater than maxlevel")
        depth = wp.maxlevel

    wp = pywt.WaveletPacket2D(data=data_g, wavelet=wavelet, mode='symmetric',maxlevel=depth)
    new_wp = pywt.WaveletPacket2D(data=np.zeros_like(data), wavelet=wavelet, mode='symmetric',maxlevel=depth)

    for n in range(0,depth):
        paths = [node.path for node in wp.get_level(n)]
        for i,path in enumerate(paths):
            new_wp[path] = wp[path].data

    paths = [n.path for n in wp.get_level(depth)]
    leveldata = [n.data for n in wp.get_level(depth)]
    print(np.shape(leveldata[1:]))
    #levelthreshold = np.percentile(np.abs(leveldata[1:]),100*threshold)
    levelthreshold = np.percentile(np.abs(leveldata[1:]), 100)*threshold
    print("levelthreshold: ",levelthreshold)
    for i,path in enumerate(paths):
        if i == 0:
            print(path)
            new_wp[path] = wp[path].data
        else:
            new_wp[path] = pywt.threshold(wp[path].data, levelthreshold, mode=thresholdmode, substitute=0)
    leveldatanew = [node.data for node in new_wp.get_level(depth)]
    print(100*(1-(np.count_nonzero(leveldatanew)/np.size(leveldatanew))))

    if plot:
        plt.figure()
        plt.xscale('log')
        if plotlog:
            plt.yscale('log')
        cmap = plt.get_cmap('tab10')
        colour = 0

        plt.axhline(y=levelthreshold, color='black', linestyle='--')
        flatcoeff = np.array(leveldatanew[1:]).flatten()
        flatoldcoeff = np.array(leveldata[1:]).flatten()

        plt.plot(sorted(np.abs(flatcoeff), reverse=True), color='black')
        plt.plot(sorted(np.abs(flatoldcoeff), reverse=True), color='black', linestyle=':')

        plt.ylabel("Amplitude of coefficients")
        plt.xlabel("Coefficients sorted by amplitude")

        plt.savefig('waveletpacketcoeff_' + str(wavelet) + '_' + str(depth) + str(threshold) + str(thresholdmode) + '.png')

    y_smooth = new_wp.reconstruct(update=True)
    y_fluc = data-y_smooth
    y_fluc_norm = y_fluc/y_smooth
    return y_fluc, y_fluc_norm, y_smooth

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
    y_fluc_norm = y_fluc/smooth

    return y_fluc, y_fluc_norm, smooth

def get_fluctuations_wavelet_undec_radial(data, scales, elliptical_model_vals, pixsize, NAXIS1, NAXIS2, image_length, wavelet='BsplineWaveletTransformATrousAlgorithm', data_g=0, smoothlvl=1, noiselvl=0):
    import pysap
    if data_g.all() == 0:
        data_g = data
    transform_klass = pysap.load_transform(wavelet)
    transform = transform_klass(nb_scale=scales, verbose=1, padding_mode="symmetric")
    transform.data = data_g
    transform.analysis()
    smoothscale = []
    print(np.shape(transform.analysis_data))
    for n in range(scales-smoothlvl,scales):
        print(n)
        smoothscale.append(get_fluctuations_ellipse(np.pad(transform.analysis_data[n],280), elliptical_model_vals, pixsize, NAXIS1, NAXIS2, image_length, data_g=np.pad(transform.analysis_data[n],280))[2])
    smooth = np.sum(smoothscale,axis=0)
    y_fluc = data-smooth

    if noiselvl != 0:
        transform1 = transform_klass(nb_scale=scales, verbose=1, padding_mode="symmetric")
        transform1.data = data
        transform1.analysis()
        noise = np.sum(transform.analysis_data[:noiselvl],axis=0)
        y_fluc = y_fluc-noise
    y_fluc_norm = y_fluc/smooth

    return y_fluc, y_fluc_norm, smooth

def plot_wavelet_undec(data, scale, wavelet='BsplineWaveletTransformATrousAlgorithm',data_g=0):
    import pysap
    if data_g.all() == 0:
        data_g = data
    transform_klass = pysap.load_transform(wavelet)
    transform = transform_klass(nb_scale=scale, verbose=1, padding_mode="symmetric")
    transform.data = data_g
    transform.analysis()
    print(scale)
    if scale % 2 == 0:
        fig, axs = plt.subplots(2, int(scale/2))
        print("even")
        cut = int(scale/2)
    else:
        fig, axs = plt.subplots(2, int((scale+1) / 2))
        print("odd")
        cut = int((scale+1) / 2)
    for n in range(0,cut*2):
        if n < cut:
            axs[0,n].imshow(transform.analysis_data[n])
            axs[0,n].get_yaxis().set_visible(False)
            axs[0,n].get_xaxis().set_visible(False)
        else:
            if n < scale:
                axs[1,int(n-cut)].imshow(transform.analysis_data[n])
            axs[1,int(n-cut)].get_yaxis().set_visible(False)
            axs[1,int(n-cut)].get_xaxis().set_visible(False)

    plt.savefig('waveletundec_decomp_'+str(wavelet)+'_'+str(scale)+'.png')
    return transform.analysis_data

def plot_wavelet_coeff(data, scale, wavelet='db3',data_g=0,threshold=0.04,log=True):
    import pysap
    if data_g.all() == 0:
        data_g = data
    import cmasher as cmr
    transform_klass = pysap.load_transform(wavelet)
    transform = transform_klass(nb_scale=scale, verbose=1, padding_mode="symmetric")
    transform.data = data_g
    transform.analysis()
    plt.figure()
    plt.xscale('log')
    if log:
        plt.yscale('log')
    colors = cmr.take_cmap_colors('cmr.ocean', scale*3, cmap_range=(0.15, 0.85), return_fmt='hex')
    colour = 0
    for n in range(0,scale*3+1):
        if n == 0:
            m = np.max(np.max(abs(transform.analysis_data[0]))) * threshold
        else:
            if n % 3 == 1:
                m = np.max((np.max(abs(transform.analysis_data[n])),np.max(abs(transform.analysis_data[n+1])),np.max(abs(transform.analysis_data[n+2]))))*threshold
            if n % 3 == 2:
                m = np.max((np.max(abs(transform.analysis_data[n-1])), np.max(abs(transform.analysis_data[n])), np.max(abs(transform.analysis_data[n + 1])))) * threshold
            if n % 3 == 0:
                m = np.max((np.max(abs(transform.analysis_data[n])), np.max(abs(transform.analysis_data[n - 1])),np.max(abs(transform.analysis_data[n - 2])))) * threshold

        plt.axhline(y=m, color=colors[colour], linestyle='--')
        plt.plot(sorted(abs(transform.analysis_data[n].flatten()),reverse=True), color=colors[colour],label="scale="+str(n))
        colour += 1
    plt.ylabel("Amplitude of coefficients")
    plt.xlabel("Coefficients, sorted by amplitude")
    plt.legend()
    plt.savefig('waveletcoeff_'+str(wavelet)+'_'+str(scale)+str(threshold)+'final.png')
    return transform.analysis_data

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

def namaster_spectrum(norm_y_fluc, bin_number, minscale, maxscale, arcmin2kpc, pixsize, theta_ap, cr, norm_y_fluc1=0):
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
    l_min = (180 * 60 * arcmin2kpc / maxscale)
    l_max = (180 * 60 * arcmin2kpc / minscale)

    bin_size = (l_max - l_min) / bin_number

    l0_bins = []
    lf_bins = []

    for i in range(bin_number):
        l0_bins.append(l_min + bin_size * i)
        lf_bins.append(l_min + bin_size * (i + 1))

    #effective l's
    b = nmt.NmtBinFlat(l0_bins, lf_bins)
    ells_uncoupled = b.get_effective_ells()
    lambdas_inv = ells_uncoupled / (arcmin2kpc * 60 * 180)
    k = 2 * np.pi * lambdas_inv

    f0 = nmt.NmtFieldFlat(Lx, Ly, mask, [norm_y_fluc], beam=[ells_uncoupled, beam(ells_uncoupled)])
    f1 = nmt.NmtFieldFlat(Lx, Ly, mask, [norm_y_fluc1], beam=[ells_uncoupled, beam(ells_uncoupled)])

    ## ANGULAR POWER SPECTRUM

    w00 = nmt.NmtWorkspaceFlat()
    w00.compute_coupling_matrix(f0, f1, b)
    #Coupling matrix used to estimate angular spectrum
    cl00_coupled = nmt.compute_coupled_cell_flat(f0, f1, b)
    cl00_uncoupled = w00.decouple_cell(cl00_coupled)[0]

    amp = abs((ells_uncoupled**2)*cl00_uncoupled/(2*np.pi))**(1/2)
    #amp = abs((k**2)*cl00_uncoupled/(2*np.pi))**(1/2)
    ## Covariance matrix

    cw = nmt.NmtCovarianceWorkspaceFlat()
    cw.compute_coupling_coefficients(f0, f1, b)
    covar = nmt.gaussian_covariance_flat(cw, 0, 0, 0, 0, ells_uncoupled,
                                         [cl00_uncoupled], [cl00_uncoupled],
                                         [cl00_uncoupled], [cl00_uncoupled], w00)

    std_power = (np.diag(covar))
    std_amp = np.sqrt(abs((ells_uncoupled**2)*std_power/(2*np.pi))**(1/2))
    #std_amp = np.sqrt(abs((k**2)*std_power/(2*np.pi))**(1/2))

    return amp, std_amp, cl00_uncoupled, covar, ells_uncoupled

def namaster_spectrum_khatri(norm_y_fluc, bin_number, minscale, maxscale, arcmin2kpc, pixsize, theta_ap, cr, norm_y_fluc1=0):
    import pymaster as nmt
    khatrix, khatriy, khatrierror = (0.000461089, 0.00056334, 0.000717526, 0.000974452, 0.001333839, 0.001846432), \
                                    (2.89673913, 3.336956522, 3.798913043, 4.027173913, 3.586956522, 2.934782609), \
                                    [[0.380434783, 0.434782609, 0.505434783, 0.586956522, 0.679347826, 0.657608696],
                                     [0.380434783, 0.440217391, 0.505434783, 0.592391304, 0.684782609, 0.663043478]]
    if norm_y_fluc1.all() == 0:
        norm_y_fluc1 = norm_y_fluc

    Lx = 4. * np.pi / 180
    Ly = 4. * np.pi / 180
    Nx, Ny = np.shape(norm_y_fluc) #len(norm_y_fluc), len(norm_y_fluc)
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
    l_min = (180 * 60 * arcmin2kpc / maxscale)
    l_max = (180 * 60 * arcmin2kpc / minscale)

    bin_size = (l_max - l_min) / bin_number

    l0_bins = []
    lf_bins = []

    for i in range(bin_number):
        l0_bins.append(l_min + bin_size * i)
        lf_bins.append(l_min + bin_size * (i + 1))

    #effective l's
    b = nmt.NmtBinFlat(l0_bins, lf_bins)
    ells_uncoupled = b.get_effective_ells()

    lambdas_inv = ells_uncoupled / (arcmin2kpc * 60 * 180)
    k = 2 * np.pi * lambdas_inv

    ells_uncoupled_khatri = tuple([x * arcmin2kpc * 60 * 180 for x in khatrix])
    ells_uncoupled_khatri_mid = [10**((np.log10(s)+np.log10(t))/2) for s, t in zip(ells_uncoupled_khatri, ells_uncoupled_khatri[1:])]

    l0_bins_klow = []
    lf_bins_klow = []

    l0_bins_klow.append(2 * ells_uncoupled_khatri[0] - ells_uncoupled_khatri_mid[0])
    for i in range(5):
        l0_bins_klow.append(ells_uncoupled_khatri_mid[i])
        lf_bins_klow.append(ells_uncoupled_khatri_mid[i])
    lf_bins_klow.append(2 * ells_uncoupled_khatri[5] - ells_uncoupled_khatri_mid[4])

    # for i in range(5):
    #     if i == 0:
    #         l0_bins_klow.append(ells_uncoupled_khatri[i] - ells_uncoupled_khatri_mid[i]/2)
    #     else:
    #         l0_bins_klow.append(lf_bins_klow[i-1])
    #     lf_bins_klow.append(ells_uncoupled_khatri[i] + ells_uncoupled_khatri_mid[i]/2)
    # l0_bins_klow.append(lf_bins_klow[4])
    # lf_bins_klow.append(ells_uncoupled_khatri[5] + ells_uncoupled_khatri_mid[4] / 2)

    print(l0_bins_klow)
    print(lf_bins_klow)
    print(l0_bins)
    print(lf_bins)
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

def namaster_spectrum_manual(norm_y_fluc, lambdas_inv, minscale, maxscale, arcmin2kpc, pixsize, theta_ap, cr, norm_y_fluc1=0):
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
    l_min = (180 * 60 * arcmin2kpc / maxscale)
    l_max = (180 * 60 * arcmin2kpc / minscale)

    bin_size = (l_max - l_min) / bin_number

    l0_bins = []
    lf_bins = []

    for i in range(bin_number):
        l0_bins.append(l_min + bin_size * i)
        lf_bins.append(l_min + bin_size * (i + 1))

    #effective l's
    b = nmt.NmtBinFlat(l0_bins, lf_bins)
    ells_uncoupled = b.get_effective_ells()
    ells_uncoupled = lambdas_inv * (arcmin2kpc * 60 * 180)
    k = 2 * np.pi * lambdas_inv

    f0 = nmt.NmtFieldFlat(Lx, Ly, mask, [norm_y_fluc], beam=[ells_uncoupled, beam(ells_uncoupled)])
    f1 = nmt.NmtFieldFlat(Lx, Ly, mask, [norm_y_fluc1], beam=[ells_uncoupled, beam(ells_uncoupled)])

    ## ANGULAR POWER SPECTRUM

    w00 = nmt.NmtWorkspaceFlat()
    w00.compute_coupling_matrix(f0, f1, b)
    #Coupling matrix used to estimate angular spectrum
    cl00_coupled = nmt.compute_coupled_cell_flat(f0, f1, b)
    cl00_uncoupled = w00.decouple_cell(cl00_coupled)[0]

    amp = abs((ells_uncoupled**2)*cl00_uncoupled/(2*np.pi))**(1/2)
    #amp = abs((k**2)*cl00_uncoupled/(2*np.pi))**(1/2)
    ## Covariance matrix

    cw = nmt.NmtCovarianceWorkspaceFlat()
    cw.compute_coupling_coefficients(f0, f1, b)
    covar = nmt.gaussian_covariance_flat(cw, 0, 0, 0, 0, ells_uncoupled,
                                         [cl00_uncoupled], [cl00_uncoupled],
                                         [cl00_uncoupled], [cl00_uncoupled], w00)

    std_power = (np.diag(covar))
    std_amp = np.sqrt(abs((ells_uncoupled**2)*std_power/(2*np.pi))**(1/2))
    #std_amp = np.sqrt(abs((k**2)*std_power/(2*np.pi))**(1/2))

    return amp, std_amp, cl00_uncoupled, covar, ells_uncoupled

def plot_pressure_spectrum(ells_uncoupled, cl00_uncoupled, Ns, amp_pressure, arcmin2kpc, std_amp):
    lambdas_inv = ells_uncoupled/(arcmin2kpc*60*180)
    k = 2*np.pi*lambdas_inv                      
    for i in range(500,2000):
        amp_pressure[i-500,:] = abs((ells_uncoupled**2)*cl00_uncoupled*k/(2*np.pi**2*Ns[i]))**(1/2)

    plt.figure()
    plt.errorbar(lambdas_inv,amp_pressure[0], yerr=std_amp*(k/Ns[0]/np.pi)**(1/2), fmt='r.',ecolor='black',elinewidth=1,
            capsize = 4,label="theta = 500 kpc")
    plt.errorbar(lambdas_inv-1e-5,amp_pressure[1499], yerr=std_amp*(k/Ns[1499]/np.pi)**(1/2), fmt='.',ecolor='black',elinewidth=1,
            capsize = 4, label='theta = 2000 kpc')
    plt.xlabel("$1/\lambda$ ($kpc^{-1}$)")
    plt.ylabel("Amplitude of pressure power spectrum")
    plt.legend()
    plt.title("Pressure Power Spectrum of Coma")
    plt.plot()
    plt.show()    
    
def radial_profile(data, center):
    y, x = np.indices((data.shape))
    r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    r = r.astype(int)

    tbin = np.bincount(r.ravel(), data.ravel())
    nr = np.bincount(r.ravel())
    radialprofile = tbin / nr
    return radialprofile

def radial_profilestat(data, center):
    y, x = np.indices((data.shape))
    r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    r = r.astype(int)


    radialprofile = scipy.stats.binned_statistic(r.ravel(), data.ravel(), statistic = 'mean', bins = np.arange(np.min(r),np.max(r)))
    radialstd = scipy.stats.binned_statistic(r.ravel(), data.ravel(), statistic = 'std', bins = np.arange(np.min(r),np.max(r)))
    return radialprofile, radialstd

def plot_y_fluc(y_fluc):
    x_ticks = ['-2', '-1','0','1','2']
    y_ticks = ['-2', '-1','0','1','2']
    t11 = [0,35,70,105,138]
    plt.figure()
    #plt.xticks(ticks=t11, labels=x_ticks, size='small')
    #plt.yticks(ticks=t11, labels=y_ticks, size='small')
    norm = TwoSlopeNorm(vmin=np.maximum(-1.5,y_fluc.min()), vcenter=0, vmax=np.minimum(1.5,y_fluc.max()))
    pc = plt.pcolormesh(y_fluc, norm=norm, cmap="seismic")
    plt.imshow(y_fluc, cmap = 'seismic')
    ax = plt.gca()
    plt.show()
   