'''
Author: Arrykrishna Mootoovaloo
Collaborators: Alan Heavens, Andrew Jaffe, Florent Leclercq
Email : a.mootoovaloo17@imperial.ac.uk
Affiliation : Imperial Centre for Inference and Cosmology
Status : Under Development
Description : Function to train the Gaussian Processes

We fix some parameters here (see settings file)
'''

import types
import importlib.machinery as im
import pandas as pd
import numpy as np
import emulator.semigp as emu_sgp
import emulator.glm as emu_glm


def training(file_settings, x_train=None, y_train=None):
    '''
    Function for training the Gaussian Processes

    Inputs
    ------
    x_train (np.ndarray) : the inputs to the GP

    y_train (np.ndarray) : the output from the traning point

    Returns
    -------
    gp_module (class) : Python class with the trained Gaussian Process
    '''

    # Load settings file
    loader = im.SourceFileLoader(file_settings, file_settings)
    settings = types.ModuleType(loader.name)
    loader.exec_module(settings)

    # ------------------------------------------------------------------------
    # instantiate the GLM module
    glm_module = emu_glm.GLM(
        theta=x_train,
        y=y_train,
        order=settings.order,
        var=settings.var,
        x_trans=settings.x_trans,
        y_trans=settings.y_trans,
        use_mean=settings.use_mean)

    # make the appropriate transformation
    # rotation of the input parameters
    glm_module.do_transformation()

    # compute the basis functions
    phi = glm_module.compute_basis()

    # set the regression prior
    glm_module.regression_prior(lambda_cap=settings.lambda_cap)

    # compute the log_evidence
    log_evi = glm_module.evidence()

    # calculate the posterior mean and variance of the regression coefficients
    post_beta, cov_beta = glm_module.posterior_coefficients()

    # ------------------------------------------------------------------------

    # number of kernel hyperparameters (amplitude and 7 lengthscales)
    n_params = int(x_train.shape[1] + 1)

    # instantiate the GP module
    gp_module = emu_sgp.GP(
        theta=x_train,
        y=y_train,
        var=settings.var,
        order=settings.order,
        x_trans=settings.x_trans,
        y_trans=settings.y_trans,
        jitter=settings.jitter,
        use_mean=settings.use_mean)

    # Make appropriate transformation
    gp_module.do_transformation()

    # Compute design matrix
    phi_gp = gp_module.compute_basis()

    # Input regression prior
    # (default: 0 mean and unit variance: inputs -> mean = None, cov = None, Lambda = 1)
    gp_module.regression_prior(
        mean=np.zeros(
            phi_gp.shape[1]), cov=np.identity(
            phi_gp.shape[1]), lambda_cap=settings.lambda_cap)

    # number of kernel hyperparameters
    n_params = x_train.shape[1] + 1

    # Set bound (prior for kernel hyperparameters)
    bnd = np.repeat(np.array([[settings.l_min, settings.l_max]]), n_params, axis=0)

    # amplitude of the residuals
    res = np.atleast_2d(y_train).T - np.dot(glm_module.phi, post_beta)
    res = res.flatten()
    amp = 2 * np.log(np.std(res))

    print('The amplitude is {0:.2f}'.format(amp))

    # we set a different bound for the amplitude
    # but one could use the answer from the Gaussian Linear Model
    # to propose an informative bound
    bnd[0] = np.array([settings.a_min, settings.a_max])

    # run optimisation
    gp_module.fit(
        method=settings.method,
        bounds=bnd,
        options={
            'ftol': settings.ftol,
            'maxiter': settings.maxiter},
        n_restart=settings.n_restart)

    return gp_module
