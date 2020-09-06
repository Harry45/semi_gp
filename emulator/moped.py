'''
Author: Arrykrishna Mootoovaloo
Collaborators: Alan Heavens, Andrew Jaffe, Florent Leclercq
Email : a.mootoovaloo17@imperial.ac.uk
Affiliation : Imperial Centre for Inference and Cosmology
Status : Under Development
Description : MOPED Algorithm
'''

import os
import numpy as np
from emulator.kids_likelihood import KiDSlike
import emulator.priors as pri
import emulator.utils.helpers as hp


class MOPED:
    '''
    Implementation of the MOPED algorithm

    Inputs:
            parameters (np.ndarray): parameters at which the compression is performed

            eps (float): epsilon value for computing finite differences
    '''

    def __init__(self, settings):

        # settings
        self.settings = settings

        # initialise likelihood object
        self.kids = KiDSlike(self.settings)

        # store the MOPED matrix and compressed data
        self.b_matrix = None
        self.y_alphas = None

        # number of MOPED coefficients
        self.n_moped = None

        # store the GPs
        self.gps = None

        # store the lhs method
        self.lhs_method = None

    def compression(self, eps, parameters):
        '''
        Compressing the data via MOPED (central finite difference method)

        Inputs:
                eps (float) : step size to compute gradient with finite difference

                parameters (np.ndarray): parameters at which the compression is performed

        Outputs:
                b_matrix (np.ndarray): B matrix of size ndim x ndata

                y_alphas (np.ndarray): the compressed data
        '''

        # compute inverse of the covariance matrix
        cov_inv = np.linalg.inv(self.kids.covariance)

        # dimension of the problem
        ndim = len(parameters)

        # number of data points
        n_data = len(self.kids.band_powers)

        # matrix to record the gradients
        grad = np.zeros((ndim, n_data))

        # some arrays to store important quantities
        cinv_grad_mu = np.zeros((ndim, n_data))
        grad_cinv_grad = np.zeros(ndim)
        b_matrix = np.zeros((ndim, n_data))

        # implementation of central finite difference method
        for i in range(ndim):

            parameters_plus = np.copy(parameters)
            parameters_minus = np.copy(parameters)
            parameters_plus[i] = parameters_plus[i] + eps
            parameters_minus[i] = parameters_minus[i] - eps

            cl_total_plus = self.kids.theory(parameters_plus)
            cl_total_minus = self.kids.theory(parameters_minus)
            grad[i] = (cl_total_plus - cl_total_minus) / (2.0 * eps)

            cinv_grad_mu[i] = np.dot(cov_inv, grad[i])
            grad_cinv_grad[i] = np.dot(grad[i], cinv_grad_mu[i])

        for i in range(ndim):

            if i == 0:
                b_matrix[i] = cinv_grad_mu[i] / np.sqrt(grad_cinv_grad[i])

            else:

                dummy_numerator = np.zeros((n_data, int(i)))
                dummy_denominator = np.zeros(int(i))

                for j in range(i):
                    dummy_numerator[:, j] = np.dot(grad[i], b_matrix[j]) * b_matrix[j]
                    dummy_denominator[j] = np.dot(grad[i], b_matrix[j])**2

                b_matrix[i] = (cinv_grad_mu[i] - np.sum(dummy_numerator, axis=1)) / \
                    np.sqrt(grad_cinv_grad[i] - np.sum(dummy_denominator))

        # compute compressed data
        y_alphas = np.dot(b_matrix, self.kids.band_powers)

        for i in range(ndim):
            for j in range(i + 1):
                if i == j:
                    print('Dot product between {0:2d} and {1:2d} is : {2:.4f}'.format(
                        i, j, np.dot(b_matrix[i], np.dot(self.kids.covariance, b_matrix[j]))))

        self.b_matrix = b_matrix
        self.y_alphas = y_alphas

        return b_matrix, y_alphas

    def compress_theory(self, parameters):
        '''
        Once the weight/MOPED vectors are computed, we now compute the compressed theory

        Inputs:
            parameters (np.ndarray): array of parameters

        Outputs:
            expected_y (np.ndarray): compressed theory of size ndim
        '''

        # theory in original data space
        expectation = self.kids.theory(parameters)

        # compressed data using MOPED
        expected_y = np.dot(self.b_matrix, expectation)

        return expected_y

    def save_vectors(self, folder_name):
        '''
        Save the MOPED vectors and compressed data to a file

        Inputs:
            folder_name (str): folder name

        Outputs:
            None
        '''
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        # MOPED Vectors
        np.savez_compressed(folder_name + '/B.npz', self.b_matrix)

        # MOPED Compressed Data
        np.savez_compressed(folder_name + '/y.npz', self.y_alphas)

    def load_vectors(self, folder_name):
        '''
        Load the MOPED vectors and compressed data

        Please provide the correct folder name where the compressed data
        and MOPED vectors are stored.

        Inputs:
            folder_name (str): folder name

        Outputs:
            b_matrix (np.ndarray): MOPED vectors (B matrix)
            y_alphas (np.ndarray): the compressed data
        '''

        b_matrix = np.load(folder_name + '/B.npz')['arr_0']

        y_alphas = np.load(folder_name + '/y.npz')['arr_0']

        self.b_matrix = b_matrix
        self.y_alphas = y_alphas

        # number of MOPED coefficients
        self.n_moped = len(y_alphas)

        return b_matrix, y_alphas

    def compress_theory_lensing(self, par):
        '''
        Compress the theory separately by individual power spectrum

        Inputs:
            par (np.ndarray): array of parameters

        Outputs:
            gg_comp (np.ndarray): array for the compressed theory (GG)

            gi_comp (np.ndarray): array for the compressed theory (GI)

            ii_comp (np.ndarray): array for the compressed theory (II)
        '''
        gg_band, gi_band, ii_band = self.kids.lensing_bandpowers(par)

        # use only the selected band powers
        gg_band = gg_band[self.kids.bands_ee_selected == 1]
        gi_band = gi_band[self.kids.bands_ee_selected == 1]
        ii_band = ii_band[self.kids.bands_ee_selected == 1]

        # number of points
        n_band = len(gg_band)

        # split the MOPED matrix
        b_matrix_1 = self.b_matrix[:, 0:n_band]

        # compute compressed theory
        gg_comp = np.dot(b_matrix_1, gg_band)
        gi_comp = np.dot(b_matrix_1, gi_band)
        ii_comp = np.dot(b_matrix_1, ii_band)

        return gg_comp, gi_comp, ii_comp

    def compress_theory_systematics(self, par):
        '''
        Compress the systematic part of the model

        By default, all BB bandpowers are zero (because resetting bias = 0), so we don't consider this

        Inputs:
            par (np.ndarrary): array of parameters

        Outputs:
            sys_comp (np.ndarray): array for the compressed systematics
        '''

        # compute systematics
        bb_bp, ee_bp_noisy, bb_bp_noisy = self.kids.systematics_calc(par)
        del bb_bp

        # Choose the ones which are used by KiDS-450
        ee_bp_noisy = ee_bp_noisy[self.kids.bands_ee_selected == 1]
        bb_bp_noisy = bb_bp_noisy[self.kids.bands_bb_selected == 1]

        # concatenate EE and BB noisy bandpowers
        ee_bb_noisy = np.concatenate((ee_bp_noisy, bb_bp_noisy))

        # compress the systematics part
        sys_comp = np.dot(self.b_matrix, ee_bb_noisy)

        return sys_comp

    def load_gps(self, folder_name, lhs_method):
        '''
        We will load all the trained Gaussian Processes and we will use
        them to make predictions at points in parameter space

        Inputs
        ------
        folder_name (str) : name of the folder where we store the GPs

        lhs_method (str) : the method we have generated the LHS samples, for example, optimum_500

        Returns
        -------
        gps (dictionary) : a dictionary with all the trained GPs
        '''

        # we need the number of MOPED coefficients to record all the GPs
        assert isinstance(self.n_moped, int), 'Call the function load_vectors prior to load_gps'

        if self.kids.settings.eight_dimensional:

            # create an empty list to store all GPs
            gps = []

            for i in range(self.n_moped):
                gps.append(hp.load_pkl_file(folder_name + '/' + lhs_method, 'gp_' + str(i)))

        else:

            # specify the spectra we need - this is by default
            if self.kids.settings.zero_mean_gp:

                spectra = ['zero_gg_' + lhs_method, 'zero_gi_' + lhs_method, 'zero_ii_' + lhs_method]

            else:

                spectra = ['gg_' + lhs_method, 'gi_' + lhs_method, 'ii_' + lhs_method]

            # update the lhs method
            self.lhs_method = lhs_method

            # number of different compressed data types
            n_spectra = len(spectra)

            # dictionary to store all the GPs
            gps = {}

            for j in range(n_spectra):
                gps[spectra[j]] = []

                for i in range(self.n_moped):
                    gps[spectra[j]].append(hp.load_pkl_file(folder_name + '/' + spectra[j], 'gp_' + str(i)))

        self.gps = gps

        return gps

    def theory_moped(self, par):
        '''
        Given a set of parameters, we use the expensive part and the systematic part
        to compute the (total) compressed theory

        Inputs:
            par (np.ndarray): array of parameters

            emulator (bool) : if True, the emulator will be used to calculate the compressed theory

        Outputs:
            comp (np.ndarray): array of compressed theory of size ndim
        '''

        # systematic part
        systematic_part = self.compress_theory_systematics(par)

        # we need A_IA to compute the sum
        cosmo, syst, other, neut = self.kids.dictionary_params(par)
        del cosmo, other, neut

        if self.kids.settings.emulator:

            if self.kids.settings.eight_dimensional:

                # the first 7 parameters and last parameter are inputs to the emulator
                test_point = np.concatenate([par[0:7], par[-1:]])

                comp_total = np.zeros(self.n_moped)

                for i in range(self.n_moped):
                    # we are using the mean only
                    comp_total[i] = self.gps[i].prediction(test_point, return_var=False)

                # calculate the total MOPED coefficient
                comp = comp_total + systematic_part

            else:
                # the first 7 parameters are inputs to the emulator
                test_point = par[0:7]

                gg_comp = np.zeros(self.n_moped)
                gi_comp = np.zeros(self.n_moped)
                ii_comp = np.zeros(self.n_moped)

                for i in range(self.n_moped):

                    # we are using the mean only

                    if self.kids.settings.zero_mean_gp:

                        gg_comp[i] = self.gps['zero_gg_' + self.lhs_method][i].prediction(test_point, returnvar=False)
                        gi_comp[i] = self.gps['zero_gi_' + self.lhs_method][i].prediction(test_point, returnvar=False)
                        ii_comp[i] = self.gps['zero_ii_' + self.lhs_method][i].prediction(test_point, returnvar=False)

                    else:

                        gg_comp[i] = self.gps['gg_' + self.lhs_method][i].prediction(test_point, return_var=False)
                        gi_comp[i] = self.gps['gi_' + self.lhs_method][i].prediction(test_point, return_var=False)
                        ii_comp[i] = self.gps['ii_' + self.lhs_method][i].prediction(test_point, return_var=False)

                # calculate the total MOPED coefficient
                comp = gg_comp + np.power(syst['A_IA'], 2) * ii_comp + syst['A_IA'] * gi_comp + systematic_part

        else:
            # get the expensive part (using CLASS directly)
            gg_comp, gi_comp, ii_comp = self.compress_theory_lensing(par)

            # calculate the total MOPED coefficient
            comp = gg_comp + np.power(syst['A_IA'], 2) * ii_comp + syst['A_IA'] * gi_comp + systematic_part

        return comp

    def loglike_moped(self, par):
        '''
        Calculates the MOPED log-likelihood given a set of parameters

        Inputs:
            par (np.ndarray) : set of parameters

            prior_dict (dict): dictionary for the prior

        Outputs:
            loglike (float): the log-likelhood value
        '''
        pri = [self.kids.all_priors[i].pdf(par[i]) for i in range(len(par))]
        prodpri = np.prod(pri)

        if prodpri == 0.0:
            chi2 = 2e12

        else:
            # calculate the theory
            expectation = self.theory_moped(par)

            diff = self.y_alphas - expectation

            chi2 = np.sum(diff**2)

        return -0.5 * chi2

    def forward_simulations(self, par, nsim=10):
        '''
        Given a set of parameters, we compute (nsim) MOPED coefficients
        at each point

        Inputs:
            par (np.ndarray): 1 x ndim (7 in this case) array of inputs to the emulator

            nsim (int): number of forward simulations we want to perform
        '''

        if not self.kids.settings.bootstrap_photoz_errors:
            raise NameError('Please use random sampling of the redshifts in the setting file.')

        # empty arrays for storing the compressed theory
        gg_comp_arr = np.zeros((nsim, 11))
        gi_comp_arr = np.zeros_like(gg_comp_arr)
        ii_comp_arr = np.zeros_like(gg_comp_arr)

        for i in range(nsim):
            # just to ensure that we have 7 and 4 inputs (emulator and systematics parameters)
            par_mod = np.concatenate((par, np.zeros(4)))

            # compute compressed theory
            gg_comp_arr[i], gi_comp_arr[i], ii_comp_arr[i] = self.compress_theory_lensing(par_mod)

        dict_quant = {'par': par, 'gg': gg_comp_arr, 'gi': gi_comp_arr, 'ii': ii_comp_arr}

        return dict_quant

    def simulations(self, parameters, nsim=10, **kwargs):
        '''
        Here we will iterate over a number of parameters and compute the MOPED coefficients

        Inputs:
            parameters (np.ndarray): N x ndim array of inputs to the emulator

            nsim (int): the number of forward simulations we want

        Outputs:
            record (list): list of all the forward simulations
        '''

        # number of training points
        n_train = parameters.shape[0]

        # empty list to store all dictionaries
        record = []
        for i in range(n_train):
            record.append(self.forward_simulations(parameters[i], nsim=nsim))

        if 'folder_name' in kwargs:
            folder_name = kwargs['folder_name']

            # create a directory if it does not exist
            if not os.path.exists(folder_name):
                os.makedirs(folder_name)

            # save the forward simulations
            np.savez_compressed(folder_name + '/simulations.npz', record)

        return record

    def compress_theory_total(self, par):
        '''
        In this case, we calculate the total band power, that is, we include the intrinsic alignment parameter in the output, that is, it is an 8D function.

        Inputs
        ------
        par (np.ndarray) : array of parameters

        Outputs
        -------
        total_band (np.ndarray) : the total band powers
        '''

        # get the individual compressed data/theory
        gg_comp, gi_comp, ii_comp = self.compress_theory_lensing(par)

        # we need A_IA to compute the sum
        cosmo, syst, other, neut = self.kids.dictionary_params(par)
        del cosmo, other, neut

        # calculate the total band powers (compressed)
        total_band_comp = gg_comp + np.power(syst['A_IA'], 2) * ii_comp + syst['A_IA'] * gi_comp

        return total_band_comp
