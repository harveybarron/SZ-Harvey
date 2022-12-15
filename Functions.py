import numpy as np
from astropy.io import fits

def get_2Dys_in_annuli(data, elliptical_model_vals):
    """
    params:
        -> data: fit.open()[1].data
        -> elliptical_model_vals: Array-like with 4 elements
        -> elliptical_model_vals[0]: x coordinate of the centre of the Cluster
        -> elliptical_model_vals[1]: y coordinate of the centre of the Cluster
        -> elliptical_model_vals[2]: Range:(0,1]; Skewness of an Elliptical model
        -> elliptical_model_vals[3]: Range:[0,pi]; Rotation of the major axis from the x-axis
    returns:
        -> rs: arraylike; The centre values for each bin
        -> ys: arraylike; The average ys in bins
        -> step_size: The size of each bin in arcmins
        -> elliptical_model_vals: Described above
    """

    pixsize = 1.7177432059
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

