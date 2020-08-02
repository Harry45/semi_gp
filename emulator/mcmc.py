'''
Author: Arrykrishna Mootoovaloo
Collaborators: Alan Heavens, Andrew Jaffe, Florent Leclercq
Email : a.mootoovaloo17@imperial.ac.uk
Affiliation : Imperial Centre for Inference and Cosmology
Status : Under Development
Description : MCMC Routine for sampling the poseriors with and without MOPED compression
'''

import os
from collections import OrderedDict
import numpy as np
import emcee
import dill

# our scripts
from emulator.kids_likelihood import KiDSlike
from emulator.moped import MOPED
import emulator.priors as pri
import emulator.utils.helpers as hp 

class MCMC(object):

    '''
    MCMC Routine to sample the posterior for various possibilities:

        - CLASS: without MOPED, with mean n(z)
        - CLASS: with MOPED, with mean n(z)

        - CLASS: without MOPED, with n(z) samples
        - CLASS: with MOPED, with n(z) samples

        - GP Emulator, with MOPED, with mean n(z)
        - GP Emulator, with MOPED, with n(z) samples

    Inputs:

        setting (str) : setting file for the inference engine

        compression (bool) : if True, MOPED compression will be used

        emulator (bool): if True, we will use the GP emulator, this always assume compression
    '''

    def __init__(self, settings, compression=False):

        # setting file
        self.settings = settings

        # MOPED Comrpession - True or False
        self.compression = compression

        # get the MOPED Routine (Assume Compressed data are already computed and stored)
        self.moped = MOPED(self.settings)

    def loglike(self, params):
        '''
        Compute the log-likelihood given a set of parameters

        The log-liklelihood (depending on specific criterion) will be returned

        TO DO: to add log-likelihood for emulator

        Inputs:
            params (np.ndarray) : array of parameters

        Outputs:
            logL (float) : value of the log-likelihood
        '''
        if self.compression:
            logL = self.moped.loglike_moped(params)

        else:
            logL = self.moped.kids.loglikelihood(params)

        return logL

    def logpost(self, params):
        '''
        Calculates the log-posterior given a set of parameters

        Inputs:
            params (np.ndarray) : array of parameters

        Outputs:
            logP (float) : the log-posterior
        '''

        # calculate the log-likelihood
        logL = self.loglike(params)

        # calculate the log prior
        pri = [self.moped.kids.all_priors[i].logpdf(params[i]) for i in range(len(params))]

        # calculate the total log-prior
        log_prior = np.sum(pri)

        # calculate log-posterior
        log_posterior = logL + log_prior

        # sometimes, parameter is outside parameter space and log_prior might be NaN
        if np.isnan(log_posterior) or np.isinf(log_posterior):
            log_posterior = -1E32

        return log_posterior

    def posterior_sampling(self, starting_point, sampler_name=None):
        '''
        Perform posterior sampling

        Arguments:
            starting_point (np.ndarray) : the starting point for sampling

            sampler_name (str): if sa sampler name has been specified, the samples will be saved in the samples/ folder

        Returns:
            sampler (EMCEE module) :
        '''

        # get the step size from the setting file
        eps = np.array(self.moped.kids.settings.eps)

        # get the number of walkers from the setting file
        nwalkers = self.moped.kids.settings.n_walkers

        # get the number of samples per walker from the setting file
        nSamples = self.moped.kids.settings.n_samples

        # the number of dimension
        ndim = len(starting_point)

        # perturb the intitial position
        pos = [starting_point + eps * np.random.randn(ndim) for i in range(nwalkers)]

        sampler = emcee.EnsembleSampler(nwalkers, ndim, self.logpost)

        sampler.run_mcmc(pos, nSamples)

        if self.moped.kids.settings.emulator:
            # if we are using the emulator - delete the GPs found in the sampler
            del self.moped.gps

        if sampler_name:
            hp.store_pkl_file(sampler, 'samples', sampler_name)

        return sampler