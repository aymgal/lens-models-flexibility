# to use with the 'lens_model_rebuttal_plot' notebook

import numpy as np

from lenstronomy.LensModel.lens_model_extensions import LensModelExtensions
from lenstronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver
from lenstronomy.Analysis.lens_properties import LensProp
from lenstronomy.Util import util
import lenstronomy.Util.mask as mask_util


def get_theta_E_eff(lens_model, kwargs_lens):
    model_ext = LensModelExtensions(lens_model)
    return model_ext.effective_einstein_radius(kwargs_lens, verbose=False)

def get_gamma_eff(lens_model, kwargs_lens):
    model_ext = LensModelExtensions(lens_model)
    return model_ext.profile_slope(kwargs_lens, verbose=False)

def get_mean_kappa_annulus(lens_model, kwargs_lens, kwargs_ps, k=None, spacing=500):
    """returns the mean convergence as defined in Kochanek (2002)"""
    if len(kwargs_ps[0]['ra_image']) > 2:
        raise NotImplementedError("Only doubles are supported here")
    r_image_1 = np.sqrt(kwargs_ps[0]['ra_image'][0]**2 + kwargs_ps[0]['dec_image'][0]**2)
    r_image_2 = np.sqrt(kwargs_ps[0]['ra_image'][1]**2 + kwargs_ps[0]['dec_image'][1]**2)
    r_in  = min(r_image_1, r_image_2)
    r_out = max(r_image_1, r_image_2)
    return _mean_kappa_annulus(lens_model, kwargs_lens, r_in, r_out, k=k, spacing=spacing)
    
def _mask_annulus_2d(center_x, center_y, r_in, r_out, x, y):
    x_shift = x - center_x
    y_shift = y - center_y
    R = np.sqrt(x_shift*x_shift + y_shift*y_shift)
    mask = np.zeros_like(R)
    mask[(r_in < R) & (R < r_out)] = 1
    n = int(np.sqrt(len(x)))
    mask_2d = mask.reshape(n, n)
    return mask_2d

def _mean_kappa_annulus(lens_model, kwargs_lens, r_in, r_out, k=None, spacing=500):
    numPix = 200
    deltaPix = 0.05
    x_grid, y_grid = util.make_grid(numPix=numPix, deltapix=deltaPix)
    center_x, center_y = kwargs_lens[0]['center_x'], kwargs_lens[0]['center_y']
    x_grid += center_x
    y_grid += center_y
    kappa = lens_model.kappa(x_grid, y_grid, kwargs_lens, k=k)
    kappa = util.array2image(kappa)
    mask_annulus = _mask_annulus_2d(center_x, center_y, r_in, r_out, x_grid, y_grid)
    return np.mean(kappa*mask_annulus)

def get_kappa_on_scaled_grid(xgrid_1d, ygrid_1d, lens_model, kwargs_lens, k=None):
    thetaE = get_theta_E_eff(lens_model, kwargs_lens)
    gamma = get_gamma_eff(lens_model, kwargs_lens)
    #print(thetaE, gamma)
    xgrid_1d_scaled = xgrid_1d/thetaE*2
    ygrid_1d_scaled = ygrid_1d/thetaE*2
    kappa = lens_model.kappa(xgrid_1d_scaled, ygrid_1d_scaled, kwargs_lens, k=k)
    return util.array2image(xgrid_1d_scaled), util.array2image(ygrid_1d_scaled), util.array2image(kappa)

def get_alpha_on_scaled_grid(xgrid_1d, ygrid_1d, lens_model, kwargs_lens, k=None):
    thetaE = get_theta_E_eff(lens_model, kwargs_lens)
    gamma = get_gamma_eff(lens_model, kwargs_lens)
    #print(thetaE, gamma)
    xgrid_1d_scaled = xgrid_1d/thetaE*2
    ygrid_1d_scaled = ygrid_1d/thetaE*2
    alphax, alphay = lens_model.alpha(xgrid_1d_scaled, ygrid_1d_scaled, kwargs_lens, k=k)
    alpha = np.sqrt(alphax**2 + alphay**2)
    return util.array2image(xgrid_1d_scaled), util.array2image(ygrid_1d_scaled), util.array2image(alpha)

def get_image_positions(lens_model, kwargs_lens, xsource, ysource):
    solver = LensEquationSolver(lens_model)
    ximage, yimage = solver.image_position_from_source(kwargs_lens=kwargs_lens, 
                                                       sourcePos_x=xsource, sourcePos_y=ysource, 
                                                       min_distance=0.01, search_window=5, 
                                                       precision_limit=10**(-10), num_iter_max=100)
    return ximage, yimage

def get_time_delays(lens_model, kwargs_lens, zlens, zsource, xsource=None, ysource=None, kwargs_ps=None, kappa_ext=0, cosmo=None, return_kwargs_ps=False):
    if (xsource is None or ysource is None) and kwargs_ps is None:
        raise ValueError
    if kwargs_ps is None:
        ximage, yimage = get_image_positions(lens_model, kwargs_lens, xsource, ysource)
        kwargs_ps = [{'ra_image': np.array(ximage), 'dec_image': np.array(yimage)}]
    kwargs_model = {'lens_model_list': lens_model.lens_model_list, 'point_source_model_list': ['LENSED_POSITION']}
    lens_prop = LensProp(zlens, zsource, kwargs_model, cosmo=cosmo)
    travel_times = lens_prop.time_delays(kwargs_lens, kwargs_ps, kappa_ext=kappa_ext)
    tds = np.array(travel_times[0] - travel_times[1:])
    if return_kwargs_ps:
        return tds, kwargs_ps
    return tds

def get_vel_disp(lens_model, kwargs_lens, reff, rap, psffwhm, zlens, zsource, cosmo=None):
    # scale by theta_E
    thetaE = get_theta_E_eff(lens_model, kwargs_lens)
    rap *= thetaE
    reff *= thetaE
    
    kwargs_model = {'lens_model_list': lens_model.lens_model_list, 'lens_light_model_list': []}
    lens_prop = LensProp(zlens, zsource, kwargs_model, cosmo=cosmo)
    kwargs_lens_light = []  # under Hernquist approx, no input light model, only r_eff is used
    Hernquist_approx = True  # Hernquist approximation for the light
    MGE_mass = True  # multi-gaussian expansion of the mass profile
    kwargs_anisotropy = {'r_ani': 1 * reff}  # simple anistropy
    anisotropy_model = 'OsipkovMerritt'
    aperture_type = 'slit'
    kwargs_aperture = {'length': rap, 'width': rap}
    #kwargs_numerics = {'sampling_number': kwargs_kinem['num_eval'], 'interpol_grid_num': 100, 'log_integration': True, 
    #                   'min_integrate': 0.001, 'max_integrate': 10}
    vel_disp = lens_prop.velocity_dispersion_numerical(kwargs_lens, kwargs_lens_light, kwargs_anisotropy, 
                                                       kwargs_aperture, psffwhm, aperture_type, anisotropy_model, 
                                                       reff, psf_type='GAUSSIAN', MGE_mass=MGE_mass, 
                                                       Hernquist_approx=True)
    return float(vel_disp)

