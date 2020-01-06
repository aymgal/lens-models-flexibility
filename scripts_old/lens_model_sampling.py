# to use with the 'lens_model_rebuttal_plot' notebook

import numpy as np
import time

from lenstronomy.Sampling.parameters import Param
from cosmoHammer import ParticleSwarmOptimizer
from cosmoHammer import CosmoHammerSampler
from cosmoHammer import LikelihoodComputationChain
from cosmoHammer.util import InMemoryStorageUtil

from lens_model_observables import get_theta_E_eff, get_gamma_eff, get_time_delays, get_vel_disp


class LensModelLikelihood(object):
    """Defines the likelihod to maximize to constrain the lens model slope and Einstein radius"""
    
    def __init__(self, likelihood_list, lens_model_class, check_bounds=True, verbose=False,
                 theta_E_value=1, theta_E_error=0.01, slope_value=2, slope_error=0.01, 
                 time_delay_values=[30], time_delay_errors=[0.25], vel_disp_value=200, vel_disp_error=30, 
                 z_source=None, z_lens=None, light_r_eff=1, kinem_r_apert=1, kinem_psf_fwhm=0.6,
                 kwargs_ps=None, kwargs_lens_init=None, kwargs_fixed_lens=None, 
                 kwargs_lower_lens=None, kwargs_upper_lens=None, kwargs_sigma_lens=None):
        self.likelihood_list = likelihood_list
        self.lens_model = lens_model_class
        self.kwargs_model = {'lens_model_list': lens_model_class.lens_model_list}
        self.param = Param(self.kwargs_model, kwargs_lens_init=kwargs_lens_init, 
                           kwargs_fixed_lens=kwargs_fixed_lens, 
                           kwargs_lower_lens=kwargs_lower_lens, kwargs_upper_lens=kwargs_upper_lens)
        self.lower_limit, self.upper_limit = self.param.param_limits()
        if kwargs_lens_init is not None:
            self.init_position = self.kwargs2args(kwargs_lens_init)
        else:
            self.init_position = [{}] * len(lens_model_class.lens_model_list)
        self.sampling_sigma = self.kwargs2args(kwargs_sigma_lens)
        
        self._check_bounds  = check_bounds
        
        self._theta_E_value = theta_E_value
        self._theta_E_error = theta_E_error
        self._slope_value   = slope_value
        self._slope_error   = slope_error
        self._time_delay_values = np.array(time_delay_values)
        self._time_delay_errors = np.array(time_delay_errors)
        self._vel_disp_value   = vel_disp_value
        self._vel_disp_error   = vel_disp_error
        
        self._light_r_eff = light_r_eff
        self._kinem_r_apert = kinem_r_apert
        self._kinem_psf_fwhm = kinem_psf_fwhm
        
        self._z_lens = z_lens
        self._z_source = z_source
        self._kwargs_ps = kwargs_ps
        
        self._verbose = verbose
    
    def __call__(self, args):
        return self.logL(args)
    
    def setup(self):
        "for CosmoHammer MCMC sampler"
        pass
    
    def computeLikelihood(self, ctx):
        "for CosmoHammer MCMC sampler"
        logL, _ = self.logL(ctx.getParams())
        return logL
        
    def logL(self, args):
        if self._check_bounds:
            inside_bounds = self.check_bounds(args)
            if not inside_bounds:
                return -1e15, None
        kwargs_all = self.args2kwargs(args)
        return self.log_likelihood(kwargs_all)
    
    def log_likelihood(self, kwargs_all):
        logL = 0
        for ll_type in self.likelihood_list:
            if ll_type == 'effective_einstein_radius':
                logL += self.theta_E_log_likelihood(kwargs_all['kwargs_lens'])
            elif ll_type == 'effective_slope':
                logL += self.slope_log_likelihood(kwargs_all['kwargs_lens'])
            elif ll_type == 'time_delays':
                logL += self.time_delays_log_likelihood(kwargs_all['kwargs_lens'])
            elif ll_type == 'kinematics':
                logL += self.kinem_log_likelihood(kwargs_all['kwargs_lens'])
        return logL, None
    
    def slope_log_likelihood(self, kwargs_lens):
        slope_model = self.effective_slope(kwargs_lens)
        if not np.isfinite(slope_model):
            return -1e15
        chi2 = (slope_model - self._slope_value)**2 / self._slope_error**2 / 2
        return -chi2
    
    def theta_E_log_likelihood(self, kwargs_lens):
        theta_E_model = self.effective_einstein_radius(kwargs_lens)
        if not np.isfinite(theta_E_model):
            return -1e15
        chi2 = (theta_E_model - self._theta_E_value)**2 / self._theta_E_error**2 / 2
        return -chi2
    
    def time_delays_log_likelihood(self, kwargs_lens):
        time_delays_model = np.array(self.time_delays(kwargs_lens))
        if not np.all(np.isfinite(time_delays_model)):
            return -1e15
        chi2 = np.sum((time_delays_model - self._time_delay_values)**2 / self._time_delay_errors**2 / 2)
        return -chi2
    
    def kinem_log_likelihood(self, kwargs_lens):
        vel_disp_model = self.velocity_dispersion(kwargs_lens)
        print("HEEEEU", vel_disp_model)
        if not np.all(np.isfinite(vel_disp_model)):
            return -1e15
        chi2 = (vel_disp_model - self._vel_disp_value)**2 / self._vel_disp_error**2 / 2
        return -chi2
    
    def check_bounds(self, args):
        for i in range(len(args)):
            if not (self.lower_limit[i] < args[i] < self.upper_limit[i]):
                if self._verbose:
                    print("parameter {} with value {} hits the bounds [{}, {}] "
                          .format(i, args[i], self.lower_limit[i], self.upper_limit[i]))
                return False
        return True
    
    def effective_einstein_radius(self, kwargs_lens):
        return get_theta_E_eff(self.lens_model, kwargs_lens)
        
    def effective_slope(self, kwargs_lens):
        return get_gamma_eff(self.lens_model, kwargs_lens)
    
    def time_delays(self, kwargs_lens):
        return get_time_delays(self.lens_model, kwargs_lens, self._z_lens, self._z_source, kwargsps=self._kwargs_ps)
    
    def velocity_dispersion(self, kwargs_lens):
        return get_vel_disp(self.lens_model, kwargs_lens, self._light_r_eff, 
                            self._kinem_r_apert, self._kinem_psf_fwhm, self._z_lens, self._z_source)
    
    def args2kwargs(self, args):
        return self.param.args2kwargs(args)
    
    def kwargs2args(self, kwargs):
        return self.param.kwargs2args(kwargs_lens=kwargs)
    
    @property
    def num_param_all(self):
        #TODO
        return None
    
    @property
    def num_param_eff(self):
        return self.param.num_param()[0]
    
    @property
    def param_names(self):
        return self.param.num_param()[1]

    
class Optimizer(object):
    """Simplified version from the lenstronomy optimizer"""
    
    def __init__(self, likelihood):
        self._chain = likelihood
        self._init_pos = likelihood.init_position
        self._lower_limit, self._upper_limit = likelihood.lower_limit, likelihood.upper_limit
        print("Effective number of data points to be sampled :", likelihood.num_param_eff)
        
    def least_squares(self):
        """Optimize using the least_squares() method from SciPy"""
        fun = lambda x: self._chain(x)[0]
        x0 = np.array(self._init_pos)
        print('Solving the non-linear least squares problem...')
        time_start = time.time()
        optimize_result = scipy.optimize.least_squares(fun, x0)
        time_end = time.time()
        print("Time used for least-squares :", time_end - time_start)
        result = optimize_result.x
        logL = optimize_result.fun
        cost = optimize_result.cost
        kwargs_result = self._chain.param.args2kwargs(result)
        return kwargs_result, logL, cost
    
    def pso(self, n_particles=10, n_iterations=10, threads=1, start_in_middle=False):
        """Optimize using the Particle Swarm Optimizer from CosmoHammer"""
        lower_start = np.array(self._lower_limit)
        upper_start = np.array(self._upper_limit)
        if start_in_middle:
            init_pos = (upper_start - lower_start) / 2 + lower_start
        else:
            init_pos = self._init_pos
        
        pso = ParticleSwarmOptimizer(self._chain, lower_start, upper_start, n_particles, threads=threads)
        pso.gbest.position = init_pos
        pso.gbest.velocity = [0] * len(init_pos)
        pso.gbest.fitness, _ = self._chain(init_pos)
        
        X2_list = []
        vel_list = []
        pos_list = []
        
        time_start = time.time()
        print('Computing the PSO ...')
        num_iter = 0
        for swarm in pso.sample(n_iterations):
            X2_list.append(pso.gbest.fitness*2)
            vel_list.append(pso.gbest.velocity)
            pos_list.append(pso.gbest.position)
            num_iter += 1
            if num_iter % 10 == 0:
                print(num_iter)
        time_end = time.time()
        
        result = pso.gbest.position
        kwargs_result = self._chain.param.args2kwargs(result)
        
        print('===================')
        print("logL =", pso.gbest.fitness)
        print("Reduced X^2 of best position :", pso.gbest.fitness * 2 / (max(self._chain.num_param_eff, 1)))
        print("Effective number of data points :", self._chain.num_param_eff)
        print(kwargs_result.get('kwargs_lens', None), 'lens result')
        print("Time used for PSO :", time_end - time_start)
        print('===================')
        
        chain = [X2_list, pos_list, vel_list, []]
        param_names = self._chain.param_names
        return kwargs_result, chain, param_names

    
class Sampler(object):
    """Simplified version from the lenstronomy sampler"""
    
    def __init__(self, likelihood):
        self._chain = likelihood
        self._init_pos = likelihood.init_position
        self._lower_limit, self._upper_limit = likelihood.lower_limit, likelihood.upper_limit
        print("Effective number of data points to be sampled :", likelihood.num_param_eff)
    
    def mcmc(self, walker_ratio=10, n_burn=10, n_run=10, threads=1, kwargs_start=None):
        """Sample using the MCMC algorithm from CosmoHammer"""
        if kwargs_start is None:
            mean_start = self._chain.init_position
        else:
            mean_start = self._chain.kwargs2args(kwargs_start['kwargs_lens'])
            
        sigma_start = self._chain.sampling_sigma
        
        lower_limit, upper_limit = self._lower_limit, self._upper_limit
        mean_start = np.maximum(lower_limit, mean_start)
        mean_start = np.minimum(upper_limit, mean_start)

        low_start = mean_start - sigma_start
        high_start = mean_start + sigma_start
        low_start = np.maximum(lower_limit, low_start)
        high_start = np.minimum(upper_limit, high_start)
        sigma_start = (high_start - low_start) / 2
        mean_start = (high_start + low_start) / 2
        params = np.array([mean_start, lower_limit, upper_limit, sigma_start]).T

        chain = LikelihoodComputationChain(min=lower_limit, max=upper_limit)

        temp_dir = tempfile.mkdtemp("Hammer")
        file_prefix = os.path.join(temp_dir, "logs")
        chain.addLikelihoodModule(self._chain)
        chain.setup()

        store = InMemoryStorageUtil()
        sampler = CosmoHammerSampler(
            params=params,
            likelihoodComputationChain=chain,
            filePrefix=file_prefix,
            walkersRatio=walker_ratio,
            burninIterations=n_burn,
            sampleIterations=n_run,
            threadCount=threads,
            initPositionGenerator=None,
            storageUtil=store)
        
        time_start = time.time()
        print('Computing the MCMC...')
        sampler.startSampling()
        time_end = time.time()
        print('===================')
        print('Number of walkers :', len(mean_start)*walker_ratio)
        print('Burn-in iterations :', n_burn)
        print('Sampling iterations :', n_run)
        print("Time taken for MCMC sampling :", time_end - time_start)
        print('===================')
        try:
            shutil.rmtree(temp_dir)
        except Exception as ex:
            print(ex, 'shutil.rmtree did not work')
            pass
        param_names = self._chain.param_names
        return store.samples, store.prob, param_names
        