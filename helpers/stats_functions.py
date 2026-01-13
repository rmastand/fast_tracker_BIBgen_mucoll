import numpy as np
import matplotlib.pyplot as plt
import numdifftools
import pickle

from scipy.special import erfcinv, loggamma, erf
from scipy.optimize import curve_fit, minimize
from scipy.signal import find_peaks
from scipy import stats

from helpers.physics_functions import get_bins, get_bins_for_scan


lim_scipy =  20000
max_min =  15000

"""
CURVE FITTING
"""

def polynomial_fit(x, *theta):

    degree = len(theta) - 1
    y = np.zeros_like(x)
    for i in range(degree + 1):
        y += theta[i] * (x)**i

    return y * (y > 0) + 1e-10

# TODO for Rikab: Object Orient, should really become part of a Class.
def full_parametric_fit(x, *theta):
    return polynomial_fit(x, *theta)


n_params_DCB = 7
def DCB(x, mu, sigma, N, alpha_L, n_L, alpha_R, n_R):

    n_L = np.abs(n_L)
    n_R = np.abs(n_R)


    A_L = np.power((n_L/np.abs(alpha_L)), n_L)*np.exp(-np.abs(alpha_L)**2 / 2.0)
    A_R = np.power((n_R/np.abs(alpha_R)), n_R)*np.exp(-np.abs(alpha_R)**2 / 2.0)

    B_L = (n_L / np.abs(alpha_L)) - np.abs(alpha_L)
    B_R = (n_R / np.abs(alpha_R)) - np.abs(alpha_R)

    # normalization factor
    norm = sigma*((A_L*np.power((alpha_L + B_L),(1.0 - n_L)) / (-1.0 + n_L)) +  (A_R*np.power((alpha_R + B_R),(1.0 - n_R)) / (-1.0 + n_R)) +  np.sqrt(np.pi/2)*(erf(alpha_L/np.sqrt(2))+erf(alpha_R/np.sqrt(2))))

    y = np.zeros_like(x)
    z = (x - mu) / sigma
    # left tail
    y += np.where(z < -alpha_L, A_L*np.power((B_L - z),(-n_L)), 0)
    # right tail
    y += np.where(z > alpha_R, A_R*np.power((B_R + z), (-n_R)), 0)  
    # gaussian core
    y += np.where((z >= -alpha_L) & (z <= alpha_R), np.exp(-0.5*z**2), 0)  
    return N*y/norm

n_params_gaussian = 3
def gaussian(x, mu, sigma, N):
    
    # normalization factor
    norm =  np.sqrt(2.0*np.pi*sigma**2)

    y = np.zeros_like(x)
    y += np.exp(-0.5*(x - mu)**2/sigma**2) 
    return np.abs(N)*y/norm


def log_gaussian(x, mu, sigma, N):
    log_absN = np.log(np.abs(N))
    c = -0.5*np.log(2.0*np.pi) - np.log(sigma**2) / 2
    z = (x - mu) / sigma
    return log_absN + c - 0.5*z**2


def soft_relu(x):
    return np.maximum(1e-10, x)

def log_DCB(x, mu, sigma, N, alpha_L, n_L, alpha_R, n_R):
    n_L = np.abs(n_L)
    n_R = np.abs(n_R)
    aL = np.abs(alpha_L)
    aR = np.abs(alpha_R)

    A_L = n_L*(np.log(n_L)-np.log(aL)) - 0.5*aL**2
    A_R = n_R*(np.log(n_R)-np.log(aR)) - 0.5*aR**2
    B_L = n_L/aL - aL
    B_R = n_R/aR - aR

    if (n_L - 1.0) <= 0:
        n_L = 1.0 + 1e-10
    if (n_R - 1.0) <= 0:
        n_R = 1.0 + 1e-10


    log_tl = A_L + (1.0 - n_L)*np.log(soft_relu(aL + B_L)) - np.log(n_L - 1.0)
    log_tr = A_R + (1.0 - n_R)*np.log(soft_relu(aR + B_R)) - np.log(n_R - 1.0)
    log_tg = np.log(np.sqrt(np.pi/2.0)) + np.log(erf(aL/np.sqrt(2.0)) + erf(aR/np.sqrt(2.0)))
    sum_log_t = log_sum_exp([log_tl, log_tr, log_tg])
    log_norm = np.log(np.abs(sigma)) + sum_log_t

    
    z = (x - mu) / np.abs(sigma)
    y = np.full_like(x, -np.inf)
    mL = z < -aL
    mC = (z >= -aL) & (z <= aR)
    mR = z > aR
    y[mL] = A_L - n_L*np.log(B_L - z[mL])
    y[mC] = -0.5*z[mC]**2
    y[mR] = A_R - n_R*np.log(B_R + z[mR])
    
    return np.log(np.abs(N)) + y - log_norm


def log_sum_exp(log_vals, axis = None):
    m = np.max(log_vals)
    return m + np.log(np.sum(np.exp(log_vals - m), axis = axis))



def poly_integral(lower, upper, bin_width, *theta):

    degree = len(theta) - 1
    integral = 0
    for i in range(degree + 1):
        integral += theta[i] / (i + 1) * ((upper)**(i + 1) - (lower)**(i + 1))

    return integral / bin_width


def numeric_integral(lower, upper, bin_width, function, *theta):

    xs = np.logspace(-3, 0, 1000)
    xs = lower + (upper - lower) * xs

    ys = function(xs, *theta)
    integral = np.trapz(ys, x = xs)

    return integral  / bin_width

def calculate_chi2(y_fit, y_true, sigma):
    return np.sum((y_fit - y_true)**2 / sigma**2)


def print_dcb_params(popt, n_peaks):

    alpha_Ls = []
    alpha_Rs = []
    n_Ls = []
    n_Rs = []
    mus = []
    sigmas = []
    Ns = []

    for n_peak in range(n_peaks):
        mus.append(popt[n_peak*n_params_DCB])
        sigmas.append(popt[n_peak*n_params_DCB+1])
        Ns.append(popt[n_peak*n_params_DCB+2])
        alpha_Ls.append(popt[n_peak*n_params_DCB+3])
        n_Ls.append(popt[n_peak*n_params_DCB+4])
        alpha_Rs.append(popt[n_peak*n_params_DCB+5])
        n_Rs.append(popt[n_peak*n_params_DCB+6])

    print("Mus: ", mus)
    print("Sigmas: ", sigmas)
    print("Ns: ", Ns)
    print("Alpha_Ls: ", alpha_Ls)
    print("n_Ls: ", n_Ls)
    print("Alpha_Rs: ", alpha_Rs)
    print("n_Rs: ", n_Rs)
    print("Polynomial Coefficients: ", popt[n_peaks*n_params_DCB:])




def double_fit(function, x, y, p0, sigma = None, lower_bounds = None, upper_bounds = None, bin_weights = None, verbose = False, prior = None):


    if sigma is None:
        sigma = np.sqrt(y + 1)

    if bin_weights is None:
        bin_weights = np.ones_like(y)

    if prior is None:
        def prior(theta):
            return 0


    def likelihood(theta):
        fit_vals = function(x, *theta)
        log_likelihood = binned_likelihood(y, None, bin_weights, fit_vals)
        return -2 * (log_likelihood + prior(theta))

    # Pre-fit
    pre_fit_L = likelihood(p0)
    if verbose:
        print("Pre-Fit L: ", pre_fit_L)

    # First fit
    try:
        if lower_bounds is None:
            popt, pcov_ = curve_fit(function, x, y, p0, sigma = sigma, maxfev=lim_scipy)
        else:
            popt, pcov_ = curve_fit(function, x, y, p0, sigma = sigma, maxfev=lim_scipy, bounds = (lower_bounds, upper_bounds))
    except:
        popt = p0
        pcov_ = None

    first_fit_L = likelihood(popt)
    if verbose:
        print("First Fit L: ", first_fit_L)

    # If the first fit was bad, skip it.
    if (pre_fit_L - first_fit_L < 0.001) or np.isnan(first_fit_L):
        popt = p0
        pcov_ = None


    if lower_bounds is not None:
        bounds = [(lower_bounds[i], upper_bounds[i]) for i in range(len(lower_bounds))]
    else:
        bounds = None

    # Second fit
    fit = minimize(likelihood, x0 = popt, method = 'Nelder-Mead', options = {'maxiter': max_min}, bounds = bounds)
    popt = fit.x


    try:
        pcov = fit.hess_inv.todense()
    except:
        pcov = pcov_ 

    return fit.fun, popt, pcov


def curve_fit_m_inv(masses, fit_degree, SR_left, SR_right, plot_bins_left, plot_bins_right, plot_centers_SB, 
                    weights=None, peak_locs_scaled = None, verbose=False, p0_bkg = None):

    """
    "
    "
    Function to fit background + DCB to mass

    Returns: popt, pcov, chi2, y_vals, n_dof_fit, fit_function (if N_peaks = 0)
    "
    "
    """


    if peak_locs_scaled is None:
        N_peaks = 0
    else:
        N_peaks = len(peak_locs_scaled)


    
    if weights is None:
        weights = np.ones_like(masses)

    """
    HISTOGRAM MASSES
    """
   
    # get left SB data
    loc_bkg_left = masses[masses < SR_left]
    weights_left = weights[masses < SR_left]
    plot_centers_left = 0.5*(plot_bins_left[1:]+plot_bins_left[:-1])
    y_vals_left, y_counts_left, bins_left, bin_weights_left, bin_err_left = build_histogram(loc_bkg_left, weights_left, plot_bins_left)


    # get right SB data
    loc_bkg_right = masses[masses > SR_right]
    weights_right = weights[masses > SR_right]
    plot_centers_right = 0.5*(plot_bins_right[1:]+plot_bins_right[:-1])
    y_vals_right, y_counts_right, bins_right, bin_weights_right, bin_err_right = build_histogram(loc_bkg_right, weights_right, plot_bins_right)

    # concatenate the SB data
    y_vals = np.concatenate((y_vals_left, y_vals_right))
    bin_weights = bin_weights_left + bin_weights_right
    errs = np.concatenate((bin_err_left, bin_err_right))
    y_err = np.sqrt(errs**2 + 1)



    if N_peaks == 0: #  NO PEAKS -- JUST FIT POLYNOMIAL


        average_bin_count = np.mean(y_vals)

        # initialize p0
        p0 = [average_bin_count] + [0 for i in range(fit_degree)]
        n_params_fit = fit_degree + 1
        fit_function = polynomial_fit
        
        # set bounds for the curve_fit optimization (not really necessary for polynomial)
        lower_bounds = [-np.inf for x in range(n_params_fit)]
        upper_bounds = [np.inf for x in range(n_params_fit)]
        
        # fit the SB data with regular curvefit
        L, popt, pcov = double_fit(fit_function, plot_centers_SB, y_vals, p0, sigma = y_err, bin_weights = bin_weights, verbose = verbose)

        if verbose: 
            print(f"0 Peak L: {L}")



    elif N_peaks > 0: # N PEAKS -- FIT POLYNOMIAL + DCB

        # If p0_bkg is not provided, fit the background with just a polynomial to start
        if p0_bkg is None:
            
            L, popt, pcov, chi2_bkg, _,  n_dof_bkg, fit_function = curve_fit_m_inv(masses, fit_degree, SR_left, SR_right, plot_bins_left, 
                                                          plot_bins_right, plot_centers_SB, verbose=verbose, weights = weights)
            p0_bkg = popt

        """ FIRST DO A CURVE_FIT TO GAUSSIAN TO GET P0 FOR DCB """
         
        # define the fit function in-place with N_peaks
        def fit_function_gaussian(x, *theta):
            y = np.zeros((N_peaks+1, len(x)))
            for n_peak in range(N_peaks):
                # sum DCBs
                y[n_peak] = log_gaussian(x, theta[n_peak*n_params_gaussian], theta[n_peak*n_params_gaussian+1], theta[n_peak*n_params_gaussian+2], )
            # polynomial 
            y[N_peaks] = np.log(polynomial_fit(x, *theta[N_peaks*n_params_gaussian:]))
            return np.exp(log_sum_exp(y, axis = 0))

        # initialize p0
        # find peaks by looking for local maxima, i.e. look for bins that are greater than
                # their distance_peaks neighbors. The local maxima with the N_peaks highest counts (y values)
                # are used for the p0 initialization

        # p0_DCB contains mu, sigma, N, alpha_L, n_L, alpha_R, n_R





        p0_gauss_init = []
        for n_peak in range(N_peaks):

            peak_location = peak_locs_scaled[n_peak]

            # find the bin closest to the peak location
            bin_index = np.argmin(np.abs(plot_centers_SB - peak_location))

            bin_count = y_vals[bin_index]
            fit_count = polynomial_fit(plot_centers_SB[bin_index], *p0_bkg)
            difference = bin_count - fit_count

            initial_std = 1*1*(plot_centers_SB[1] - plot_centers_SB[0])

            initial_N = difference * np.sqrt(2 * np.pi) * initial_std


            p0_gauss_init += [peak_locs_scaled[n_peak], initial_std, initial_N]

        p0_gauss_init += list(p0_bkg)


        # Prior on gaussian parameters
        def prior(theta):

            L = 0
            std = 0.1 * (plot_centers_SB[1] - plot_centers_SB[0])
            for n_peak in range(N_peaks):
            
                mu = theta[n_peak*n_params_gaussian]
                sigma = theta[n_peak*n_params_gaussian+1]
                N = theta[n_peak*n_params_gaussian+2]

                # mu prior
                L += -0.5*(mu - peak_locs_scaled[n_peak])**2 / (std**2)

                # N prior (Dont let N be too large)
                L += -0.1 * np.log(np.abs(N)) 

            return L   
 
        # fit the SB data with regular curvefit to get an initial guess for popt
        L, p0_gaussian_final, pcov = double_fit(fit_function_gaussian, plot_centers_SB, y_vals, p0_gauss_init, sigma = y_err, bin_weights = bin_weights, verbose = verbose)
        
        popt = p0_gaussian_final
        pcov = None
        fit_function = fit_function_gaussian
        n_params_fit = len(p0_gaussian_final)

        if verbose:
            print(f"Initial Gaussian Fit L: {L}")
            # print(f"Initial Gaussian Fit Popt: {p0_gaussian_final}")
        
        popt = p0_gaussian_final
        pcov = None
        n_params_fit = len(popt)
        fit_function = fit_function_gaussian


        """ THEN DO A PROPER MINIMIZATION WITH DCB """
        
        # define the fit function in-place with N_peaks
        def fit_function_DCB(x, *theta):
            y = np.zeros((N_peaks+1, len(x)))
            for n_peak in range(N_peaks):
                # sum DCBs
                y[n_peak] = log_DCB(x, theta[n_peak*n_params_DCB], theta[n_peak*n_params_DCB+1], theta[n_peak*n_params_DCB+2], 
                         theta[n_peak*n_params_DCB+3], theta[n_peak*n_params_DCB+4], theta[n_peak*n_params_DCB+5], theta[n_peak*n_params_DCB+6])
            # polynomial 
            y[N_peaks] = np.log(polynomial_fit(x, *theta[N_peaks*n_params_DCB:]))
            return np.exp(log_sum_exp(y, axis = 0))


        p0_DCB_init = []
        for n_peak in range(N_peaks):
            # pull mu, sigma, N from the gaussian fit
            p0_DCB_init += list(p0_gaussian_final[n_peak*n_params_gaussian:(n_peak+1)*n_params_gaussian])

            # make sure the sigma is positive
            p0_DCB_init[n_peak*n_params_DCB+1] = np.abs(p0_DCB_init[n_peak*n_params_DCB+1])
            std = p0_DCB_init[n_peak*n_params_DCB+1]

            # alpha_L, n_L, alpha_R, n_R
            p0_DCB_init += [3/std, 2, 3/std, 2] 
        p0_DCB_init += list(p0_gaussian_final[n_params_gaussian*N_peaks:])
        n_params_fit = len(p0_DCB_init)


        # set bounds for the optimizations (the bound n > 1 is necessary. The others are just to help optimization)
        lower_bounds = [-np.inf, 0, -np.inf, 0, 1, 0, 1]*N_peaks + [-np.inf for x in range(fit_degree+1)]
        upper_bounds = [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf]*N_peaks + [np.inf for x in range(fit_degree+1)]
        bounds =  [(lower_bounds[i], upper_bounds[i]) for i in range(len(lower_bounds))]
        # # then the SB data with scipy minimize

        # Prior on DCB parameters
        def prior(theta):

            L = 0
            std = 0.1 * (plot_centers_SB[1] - plot_centers_SB[0])
            for n_peak in range(N_peaks):
            
                mu = theta[n_peak*n_params_gaussian]
                sigma = theta[n_peak*n_params_gaussian+1]
                N = theta[n_peak*n_params_gaussian+2]

                # mu prior
                L += -0.5*(mu - peak_locs_scaled[n_peak])**2 / (std**2)

                # N prior (Dont let N be too large)
                L += -0.1 * np.log(np.abs(N)) 

            return L   

        L, popt, pcov = double_fit(fit_function_DCB, plot_centers_SB, y_vals, p0_DCB_init, sigma = y_err, lower_bounds = lower_bounds, upper_bounds = upper_bounds, bin_weights = bin_weights, verbose = verbose)

        n_params_fit = len(popt)
        fit_function = fit_function_DCB


        if verbose:
            print(f"Final DCB Fit L: {L}")

    
    # get chi2 in the SB
    chi2 = calculate_chi2(fit_function(plot_centers_SB, *popt), y_vals, y_err)

    return L, popt, pcov, chi2, y_vals, len(y_vals) - n_params_fit, fit_function


"""
STATS
"""


def ReLU(x):
    return np.maximum(0, x)

def calculate_F_statistic_p_value(SSE_full, SSE_reduced, n_dof_full, n_dof_reduced):
    numerator = (SSE_reduced - SSE_full) / (n_dof_reduced - n_dof_full)
    denominator = SSE_full / n_dof_full
    Fstar = numerator / denominator
    p_value = 1 - stats.f.cdf(Fstar, n_dof_reduced - n_dof_full, n_dof_full)
    return Fstar, p_value 


def binned_likelihood(yvals, ycounts, weights, fit_vals):

    log_likelihood = 0


    for i in range(len(yvals)):

        expval_weights = np.mean(weights[i])
        expval_weights2 = np.mean(weights[i]**2)
        len_weights = len(weights[i])

        if len_weights == 0 or (np.abs(expval_weights - 1) < 1e-3 and np.abs(expval_weights2 - 1) < 1e-3):
            log_likelihood += stats.poisson.logpmf(yvals[i], fit_vals[i])
        
        else:


            scale_factor = expval_weights2 / expval_weights
            lambda_prime = fit_vals[i] / scale_factor

            n_prime = yvals[i] / scale_factor

            if True:
                log_likelihood += n_prime * np.log(lambda_prime) - lambda_prime - loggamma(n_prime + 1)

    return log_likelihood

def build_histogram(data, weights, bins):

    y_vals, _bins = np.histogram(data, bins = bins, density = False, weights = weights)
    y_counts, _ = np.histogram(data, bins = bins, density = False)


    digits = np.digitize(data, _bins)
    bin_weights = [weights[digits==i] for i in range(1, len(_bins))]
    bin_err = np.asarray([np.linalg.norm(weights[digits==i]) for i in range(1, len(_bins))])

    return y_vals, y_counts, bins, bin_weights, bin_err


def likelihood(data, s, BIN_INFO, weights, parametric_function, *theta):

    plot_bins_all, plot_bins_SR, plot_bins_left, plot_bins_right, plot_centers_all, plot_centers_SR, plot_centers_SB = BIN_INFO
    SR_left, SR_right = plot_bins_SR[0], plot_bins_SR[-1]
    SB_left, SB_right = plot_bins_left[0], plot_bins_right[-1]    
    plot_centers_left = 0.5*(plot_bins_left[1:] + plot_bins_left[:-1])
    plot_centers_right = 0.5*(plot_bins_right[1:] + plot_bins_right[:-1])


    if weights is None:
        weights = np.ones_like(data)

    # get left SB data
    loc_bkg_left = data[data < SR_left]
    weights_left = weights[data < SR_left]
    y_vals_left, y_counts_left, _bins, left_weights, left_err = build_histogram(loc_bkg_left, weights_left, plot_bins_left)


    # get right SB data
    loc_bkg_right = data[data > SR_right]
    weights_right = weights[data > SR_right]
    y_vals_right, y_counts_right, _bins, right_weights, right_err = build_histogram(loc_bkg_right, weights_right, plot_bins_right)

    # Log poisson likelihood for the SB bins
    log_likelihood = 0
    fit_vals_left = parametric_function(plot_centers_left, *theta)
    fit_vals_right = parametric_function(plot_centers_right, *theta)
    
    log_likelihood += binned_likelihood(y_vals_left, y_counts_left, left_weights, fit_vals_left)
    log_likelihood += binned_likelihood(y_vals_right, y_counts_right, right_weights, fit_vals_right)
      

    # get SR data
    bin_width = plot_bins_SR[1] - plot_bins_SR[0]
    loc_data = data[np.logical_and(data > SR_left, data < SR_right)]
    loc_weights = weights[np.logical_and(data > SR_left, data < SR_right)]
    num_SR = np.sum(loc_weights)
    err = np.sqrt(np.sum(loc_weights**2))

    num_bkg = numeric_integral(SR_left, SR_right, bin_width, parametric_function, *theta)
    s_prime = s * (s > 0) # Ensure positive signal. If negative, this will cancel out in the likelihood ratio

    expval_weights = np.mean(loc_weights)
    expval_weights2 = np.mean(loc_weights**2)

    # If weights are trivial
    if len(loc_weights) == 0 or (np.abs(expval_weights) < 1e-3 and np.abs(expval_weights2) < 1e-3):

        log_likelihood += stats.poisson.logpmf(num_SR, num_bkg + s_prime)

    else:
        scale_factor = expval_weights2 / expval_weights
        lambda_prime = (num_bkg + s) / scale_factor
        n_prime = num_SR / scale_factor
        log_likelihood += n_prime * np.log(lambda_prime) - lambda_prime - loggamma(n_prime + 1)
    return -2 * log_likelihood

    
def cheat_likelihood(data, BIN_INFO, weights, parametric_function, *theta):

    plot_bins_all, plot_bins_SR, plot_bins_left, plot_bins_right, plot_centers_all, plot_centers_SR, plot_centers_SB = BIN_INFO
    SR_left, SR_right = plot_bins_SR[0], plot_bins_SR[-1]
    SB_left, SB_right = plot_bins_left[0], plot_bins_right[-1]    
    plot_centers_left = 0.5*(plot_bins_left[1:] + plot_bins_left[:-1])
    plot_centers_right = 0.5*(plot_bins_right[1:] + plot_bins_right[:-1])
    

    if weights is None:
        weights = np.ones_like(data)

    # get left SB data
    loc_bkg_left = data[data < SR_left]
    weights_left = weights[data < SR_left]
    y_vals_left, y_counts_left, _bins, left_weights, left_err = build_histogram(loc_bkg_left, weights_left, plot_bins_left)

    # get right SB data
    loc_bkg_right = data[data > SR_right]
    weights_right = weights[data > SR_right]
    y_vals_right, y_counts_right, _bins, right_weights, right_err = build_histogram(loc_bkg_right, weights_right, plot_bins_right)
    
    # Log poisson likelihood for the SB bins
    log_likelihood = 0
    fit_vals_left = parametric_function(plot_centers_left, *theta)
    fit_vals_right = parametric_function(plot_centers_right, *theta)
   
    log_likelihood += binned_likelihood(y_vals_left, y_counts_left, left_weights, fit_vals_left)
    log_likelihood += binned_likelihood(y_vals_right, y_counts_right, right_weights, fit_vals_right)

    return -2 * log_likelihood
    
def null_hypothesis(data, BIN_INFO, weights, parametric_function, *theta):
    return likelihood(data, 0, BIN_INFO, weights, parametric_function, *theta)


def calculate_test_statistic(data, BIN_INFO, weights = None, parametric_function = None, degree = 5, starting_guess = None, verbose_plot = False, return_popt = False):

    # We want to determine the profiled log likelihood ratio: -2 * [L(s, theta_hat_hat) - L(s_hat, theta_hat)]
    # for s = 0

    # Set up 
    plot_bins_all, plot_bins_SR, plot_bins_left, plot_bins_right, plot_centers_all, plot_centers_SR, plot_centers_SB = BIN_INFO
    SR_left, SR_right = plot_bins_SR[0], plot_bins_SR[-1]
    SB_left, SB_right = plot_bins_left[0], plot_bins_right[-1]

    if weights is None:
        weights = np.ones_like(data)

    bin_width = plot_bins_SR[1] - plot_bins_SR[0]
    if starting_guess is None:
        average_bin_count = len(data) / len(plot_centers_all)
        starting_guess = [average_bin_count, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    if parametric_function is None:
        parametric_function = polynomial_fit

    # ########## MODIFIED PARAMETERIC FIT ##########

    num_params = len(starting_guess)
    num_params_polynomial = degree + 1
    theta_dcb = starting_guess[0:num_params - num_params_polynomial]
    starting_guess = starting_guess[num_params - num_params_polynomial:]

    def modified_parametric_function(x, *theta_polynomial):

        params = list(theta_dcb) + list(theta_polynomial)
        return parametric_function(x, *params)
        

    # Fit the s = 0 hypothesis
    lambda_null = lambda theta: null_hypothesis(data, BIN_INFO, weights, modified_parametric_function, *theta)
    fit = minimize(lambda_null , x0 = starting_guess, method = 'Nelder-Mead', options = {'maxiter': max_min, "disp": verbose_plot})
    theta_hat_hat = fit.x
    null_fit_likelihood = null_hypothesis(data, BIN_INFO, weights, modified_parametric_function, *theta_hat_hat)


    # Fit the s = float hypothesis
    lambda_cheat = lambda theta: cheat_likelihood(data, BIN_INFO, weights, modified_parametric_function, *theta)
    fit = minimize(lambda_cheat , x0 = theta_hat_hat, method = 'Nelder-Mead', options = {'maxiter': max_min, "disp": verbose_plot})
    theta_hat = fit.x

    integrated_background = numeric_integral(SR_left, SR_right, bin_width, modified_parametric_function, *theta_hat)
    loc_weights = weights[np.logical_and(data > SR_left, data < SR_right)]
    num_SR = np.sum(loc_weights)
    integrated_signal = num_SR - integrated_background
    best_fit_likelihood = likelihood(data, integrated_signal, BIN_INFO, weights, modified_parametric_function, *theta_hat)


    # Calculate the test statistic
    test_statistic = (null_fit_likelihood - best_fit_likelihood)
    if integrated_signal < 0:
        test_statistic = 0
    if test_statistic < 0:
        test_statistic = 0

    if verbose_plot:
        print('Best fit:', best_fit_likelihood)
        print('Null fit:', null_fit_likelihood)
        print('Test statistic:', null_fit_likelihood - best_fit_likelihood)
        print('Integrated signal:', integrated_signal)
        print('Integrated background:', integrated_background)

        print("Initial guess:", starting_guess)
        print("Theta hat:", theta_hat)
        print("Theta hat hat:", theta_hat_hat)


    if verbose_plot:
        plt.hist(data, bins = plot_bins_all, histtype = 'step', color = 'black', label = 'Data', weights = weights)
        plt.plot(plot_centers_all, modified_parametric_function(plot_centers_all, *theta_hat), label = 'Fit', ls = '--')
        plt.plot(plot_centers_all, modified_parametric_function(plot_centers_all, *theta_hat_hat), label = 'Null', ls = '-.')
        plt.legend()


    if return_popt:
        return integrated_signal, integrated_background, test_statistic, theta_hat

    return integrated_signal, integrated_background, test_statistic