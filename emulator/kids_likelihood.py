'''
Author: Arrykrishna Mootoovaloo
Collaborators: Alan Heavens, Andrew Jaffe, Florent Leclercq
Email : a.mootoovaloo17@imperial.ac.uk
Affiliation : Imperial Centre for Inference and Cosmology
Status : Under Development
Description : Emulator for KiDS-450 data

Acknowledgement
---------------
Original KiDS-450 likelihood code (see Kohlinger et al. 2017) from which this
code has been adapted.
'''

# pylint: disable=E1101
# pylint: disable=C0301
# pylint: disable=R0913
# pylint: disable=R0914
# pylint: disable=R0915

import os
import types
import importlib.machinery as im
import numpy as np
import scipy.interpolate as itp
from scipy import integrate
from scipy.linalg import cholesky, solve_triangular
from classy import Class
import emulator.priors as pri
# import priors as pri
import pickle


np.set_printoptions(suppress=True, formatter={'float_kind':'{:0.5f}'.format})

class KiDSlike:

    '''
    KiDSlike routine

    Arguments:
        file_settings (str): the setting file
    '''

    def __init__(self, file_settings='settings'):

        # Load settings file
        loader = im.SourceFileLoader(file_settings, file_settings)
        self.settings = types.ModuleType(loader.name)
        loader.exec_module(self.settings)

        # prior
        self.all_priors = pri.distributions(self.settings.prior)

        # =========================#
        # Some important variables #
        # =========================#

        # redshift bins min max in a list
        self.redshift_bins = []

        # number of redshift bins
        self.nzbins = None

        # total number of auto- and cross- bins
        self.nzcorrs = None

        # which ee band powers to use (from setting file)
        self.all_bands_ee_to_use = []

        # which BB band powers to use (from setting file)
        self.all_bands_bb_to_use = []

        # m-correction (use zeros or load file)
        self.m_corr_fiducial_per_zbin = None

        # indices of band powers to use
        self.indices_for_bands_to_use = None

        # standard deviation (see KiDS-450 paper)
        self.sigma_e = None

        # n_effective
        self.n_eff = None

        # covariance matrix of the data
        self.covariance = None

        # the band window matrix
        self.band_window_matrix = None

        # number of ell modes
        self.ells_intp = None

        # number of ee band powers
        self.band_offset_ee = None

        # number of BB band powers
        self.band_offset_bb = None

        # band powers
        self.band_powers = None

        # m_correction vector and matrix
        self.m_vec = None
        self.m_cov = None

        # we need redshifts and maximum redshift for the new method for computing power spectra

        self.redshift_new = None
        self.zmax_new = None
        #----------------------------------------------------------------------------------#
        # load all files
        self.load_files()

        # this is for testing - we need the redshifts anyway for the computing the power spectra
        # with the new technique
        self.redshift_new, _ , _ , self.zmax_new = self.read_redshifts()

        # Fix redshift if we are using mean
        if not self.settings.bootstrap_photoz_errors:
            self.redshifts, self.pr_red, self.pr_red_norm, self.zmax = self.read_redshifts()

        # calculates or load the m-correction
        if not self.settings.m_correction:
            self.m_vec, self.m_cov = self.calc_m_correction()

        # given the m-correction, calculate the scaled bandpowers and covariance
        self.band_powers, self.covariance, self.chol_factor = self.bandpowers_and_cov()
        #----------------------------------------------------------------------------------#

        # other important quantities required
        self.ells_min = self.ells_intp[0]
        self.ells_max = self.ells_intp[-1]
        self.nells = int(self.ells_max - self.ells_min + 1)

        # these are the \ell modes
        self.ells_sum = np.linspace(self.ells_min, self.ells_max, self.nells)

        # these are the l-nodes for the derivation of the theoretical cl:
        # \ells in logspace
        self.ells = np.logspace(np.log10(self.ells_min), np.log10(self.ells_max), self.settings.nellsmax)

        # normalisation factor
        self.ell_norm = self.ells_sum * (self.ells_sum + 1) / (2. * np.pi)

        # bands selected (indices only)
        self.bands_ee_selected = np.tile(self.settings.bands_EE_to_use, self.nzcorrs)
        self.bands_bb_selected = np.tile(self.settings.bands_BB_to_use, self.nzcorrs)

    def load_files(self):
        '''
        Load all important files
        '''
        for index_zbin in range(len(self.settings.zbin_min)):
            redshift_bin = '{:.2f}z{:.2f}'.format(
                self.settings.zbin_min[index_zbin],
                self.settings.zbin_max[index_zbin])
            self.redshift_bins.append(redshift_bin)

        # number of z-bins
        self.nzbins = len(self.redshift_bins)

        # number of *unique* correlations between z-bins
        self.nzcorrs = int(self.nzbins * (self.nzbins + 1) / 2)

        # default, use all correlations:
        for _ in range(self.nzcorrs):
            self.all_bands_ee_to_use += self.settings.bands_EE_to_use
            self.all_bands_bb_to_use += self.settings.bands_BB_to_use

        self.all_bands_ee_to_use = np.array(self.all_bands_ee_to_use)
        self.all_bands_bb_to_use = np.array(self.all_bands_bb_to_use)

        all_bands_to_use = np.concatenate((self.all_bands_ee_to_use, self.all_bands_bb_to_use))
        self.indices_for_bands_to_use = np.where(np.asarray(all_bands_to_use) == 1)[0]

        # this is also the number of points in the datavector
        # ndata = len(self.indices_for_bands_to_use)

        # m_correction average is used for MOPED
        # If file isn ot found, it is set to zero
        try:
            fname = os.path.join(self.settings.data_directory, '{:}zbins/m_correction_avg.txt'.format(self.nzbins))

            if self.nzbins == 1:
                self.m_corr_fiducial_per_zbin = np.asarray([np.loadtxt(fname, usecols=[1])])
            else:
                self.m_corr_fiducial_per_zbin = np.loadtxt(fname, usecols=[1])
        except BaseException:
            self.m_corr_fiducial_per_zbin = np.zeros(self.nzbins)
            print('Could not load m-correction values from {}\n'.format(fname))
            print('Setting them to zero instead.')

        try:
            fname = os.path.join(self.settings.data_directory,
                                 '{:}zbins/sigma_int_n_eff_{:}zbins.dat'.format(self.nzbins, self.nzbins))
            tbdata = np.loadtxt(fname)
            if self.nzbins == 1:
                # correct columns for file!
                sigma_e1 = np.asarray([tbdata[2]])
                sigma_e2 = np.asarray([tbdata[3]])
                n_eff = np.asarray([tbdata[4]])
            else:
                # correct columns for file!
                sigma_e1 = tbdata[:, 2]
                sigma_e2 = tbdata[:, 3]
                n_eff = tbdata[:, 4]

            self.sigma_e = np.sqrt((sigma_e1**2 + sigma_e2**2) / 2.)
            # convert from 1 / sq. arcmin to 1 / sterad
            self.n_eff = n_eff / np.deg2rad(1. / 60.)**2

        except BaseException:
            # these dummies will set noise power always to 0!
            self.sigma_e = np.zeros(self.nzbins)
            self.n_eff = np.ones(self.nzbins)
            print('Could not load sigma_e and n_eff!')

        # load the band powers (ee and BB)
        collect_bp_ee_in_zbins = []
        collect_bp_bb_in_zbins = []
        # collect BP per zbin and combine into one array
        for zbin1 in range(self.nzbins):
            for zbin2 in range(zbin1 + 1):
                fname_ee = os.path.join(self.settings.data_directory,
                                        '{:}zbins/band_powers_EE_z{:}xz{:}.dat'.format(self.nzbins, zbin1 + 1, zbin2 + 1))
                fname_bb = os.path.join(self.settings.data_directory,
                                        '{:}zbins/band_powers_BB_z{:}xz{:}.dat'.format(self.nzbins, zbin1 + 1, zbin2 + 1))
                extracted_band_powers_ee = np.loadtxt(fname_ee)
                extracted_band_powers_bb = np.loadtxt(fname_bb)
                collect_bp_ee_in_zbins.append(extracted_band_powers_ee)
                collect_bp_bb_in_zbins.append(extracted_band_powers_bb)

        # band powers
        self.band_powers = np.concatenate(
            (np.asarray(collect_bp_ee_in_zbins).flatten(),
             np.asarray(collect_bp_bb_in_zbins).flatten()))

        # Load the covariance matrix
        fname = os.path.join(self.settings.data_directory, '{:}zbins/covariance_all_z_EE_BB.dat'.format(self.nzbins))
        self.covariance = np.loadtxt(fname)

        # load the band window matrix
        fname = os.path.join(self.settings.data_directory,
                             '{:}zbins/band_window_matrix_nell100.dat'.format(self.nzbins))
        self.band_window_matrix = np.loadtxt(fname)

        # ells_intp and also band_offset are consistent between different patches!
        fname = os.path.join(self.settings.data_directory,
                             '{:}zbins/multipole_nodes_for_band_window_functions_nell100.dat'.format(self.nzbins))
        self.ells_intp = np.loadtxt(fname)

        self.band_offset_ee = len(extracted_band_powers_ee)
        self.band_offset_bb = len(extracted_band_powers_bb)

    @staticmethod
    def dictionary_params(par):
        '''
        A dictionary for storing all the parameters

        Inputs:
            par (np.ndarray): parameters for the inference

        Outputs:
            cosmo_params (dict): dictionary for all the cosmological parameters

            systematics (dict): dictionary for the systematic parameters

        The parameters are organised in the following order:

        -------------------
        0: omega_cdm_h2
        1: omega_b_h2
        2: ln_10_10_A_s
        3: n_s
        4: h
        5: A_bary
        6: sum_neutrino
        -------------------
        7: A_1
        8: A_2
        9: A_3
        10: A_IA
        -------------------
        '''
        cosmo_params = {'omega_cdm': par[0], 'omega_b': par[1], 'ln10^{10}A_s': par[2], 'n_s': par[3], 'h': par[4]}
        other_settings = {'N_ncdm': 1.0, 'deg_ncdm': 3.0, 'T_ncdm': 0.71611, 'N_ur': 0.00641}
        neutrino_settings = {'m_ncdm': par[6] / other_settings['deg_ncdm']}
        systematics = {'A_bary': par[5], 'A_n1': par[7], 'A_n2': par[8], 'A_n3': par[9], 'A_IA': par[10]}

        return cosmo_params, systematics, other_settings, neutrino_settings

    def matter_power_spectrum(self, par, redshifts, class_arguments):
        '''
        Inputs:
            par (np.ndarray): array of parameters for inputs to ClASS

            redshift (np.ndarray): array of redshifts at which the power spectrum will be calculated

            See dictionary_params for further details

        Outputs:
            pk (np.ndarray): matter power spectrum

            quant (dict): dictionary with important quantities
            - omega_m
            - small_h
            - chi
            - dzdr
        '''

        cosmo_par, syst, other, neutrino = KiDSlike.dictionary_params(par)

        cosmo = Class()

        # input cosmologies
        cosmo.set(cosmo_par)

        # settings for clASS (halofit etc)
        cosmo.set(class_arguments)

        # other settings for neutrino
        cosmo.set(other)

        # neutrino settings
        cosmo.set(neutrino)

        # compute power spectrum
        cosmo.compute()

        # Omega_matter
        omega_m = cosmo.Omega_m()

        # h parameter
        small_h = cosmo.h()

        # critical density
        rho_crit = KiDSlike.get_critical_density(small_h)

        # derive the linear growth factor D(z)
        linear_growth_rate = np.zeros_like(redshifts)

        for index_z, red in enumerate(redshifts):

            # compute linear growth rate
            linear_growth_rate[index_z] = cosmo.scale_independent_growth_factor(red)

            # normalise linear growth rate at redshift = 0
            linear_growth_rate /= cosmo.scale_independent_growth_factor(0.)

        # get distances from cosmo-module
        chi, dzdr = cosmo.z_of_r(redshifts)

        # Get power spectrum P(k=l/r,z(r)) from cosmological module
        pk_matter = np.zeros((self.settings.nellsmax, self.settings.nzmax + 1), 'float64')
        k_max_in_inv_mpc = self.settings.k_max_h_by_Mpc * small_h

        for index_ells in range(self.settings.nellsmax):
            for index_z in range(1, self.settings.nzmax + 1):

                # standard Limber approximation:
                # k = ells[index_ells] / r[index_z]
                # extended Limber approximation (LoVerde & Afshordi 2008):

                k_in_inv_mpc = (self.ells[index_ells] + 0.5) / chi[index_z]
                if k_in_inv_mpc > k_max_in_inv_mpc:
                    pk_dm = 0.
                else:
                    pk_dm = cosmo.pk(k_in_inv_mpc, redshifts[index_z])

                # include baryon feedback by default
                pk_matter[index_ells, index_z] = pk_dm * \
                    self.baryon_feedback_bias_sqr(k_in_inv_mpc / small_h, redshifts[index_z], a_bary=syst['A_bary'])

        # dictionary for important quantities
        dict_quantities = {
            'omega_m': omega_m,
            'small_h': small_h,
            'chi': chi,
            'dzdr': dzdr,
            'lgr': linear_growth_rate,
            'rho_crit': rho_crit}

        # clean cosmo-module to free up memory
        cosmo.struct_cleanup()
        cosmo.empty()
        del cosmo

        return pk_matter, dict_quantities

    def lensing_bandpowers(self, par):
        '''
        Inputs:
            matter_pk (np.ndarray): the matter power spectrum

        Outputs:
            ee_bp (np.ndarray): the ee bandpowers

            gi_bp (np.ndarray): the gi bandpowers

            ii_bp (np.ndarray): the ii bandpowers
        '''

        # if we want redshifts samples
        if self.settings.bootstrap_photoz_errors:
            self.redshifts, self.pr_red, self.pr_red_norm, self.zmax = self.read_redshifts()

        # CLASS arguments
        if self.settings.mode == 'halofit':
            class_arguments = {
                'z_max_pk': self.zmax,
                'output': 'mPk',
                'non linear': self.settings.mode,
                'P_k_max_h/Mpc': self.settings.k_max_h_by_Mpc}
        else:
            class_arguments = {
                'z_max_pk': self.zmax,
                'output': 'mPk',
                'P_k_max_h/Mpc': self.settings.k_max_h_by_Mpc}

        # get matter power spectrum and important quantities
        pk_matter, quant = self.matter_power_spectrum(par, self.redshifts, class_arguments)

        # if we have NaNs in the power spectrum - return an array of NaNs
        if np.isnan(pk_matter.flatten()).any():
            to_return = np.array([np.nan]*self.nzcorrs*self.band_offset_ee)
            return to_return, to_return, to_return

        else:

            # n(z) to n(chi)
            pr_chi = self.pr_red * (quant['dzdr'][:, np.newaxis] / self.pr_red_norm)

            kernel = np.zeros((self.settings.nzmax + 1, self.nzbins), 'float64')

            for zbin in range(self.nzbins):
                # assumes that z[0] = 0
                for index_z in range(1, self.settings.nzmax + 1):
                    fun = pr_chi[index_z:, zbin] * (quant['chi'][index_z:] - quant['chi'][index_z]) / quant['chi'][index_z:]
                    kernel[index_z, zbin] = np.sum(0.5 * (fun[1:] + fun[:-1]) * (quant['chi']
                                                                                 [index_z + 1:] - quant['chi'][index_z:-1]))
                    kernel[index_z, zbin] *= 2. * quant['chi'][index_z] * (1. + self.redshifts[index_z])

            # Start loop over l for computation of C_l^shear
            cl_gg_integrand = np.zeros((self.settings.nzmax + 1, self.nzbins, self.nzbins), 'float64')
            cl_ii_integrand = np.zeros_like(cl_gg_integrand)
            cl_gi_integrand = np.zeros_like(cl_gg_integrand)

            cl_gg = np.zeros((self.settings.nellsmax, self.nzbins, self.nzbins), 'float64')
            cl_ii = np.zeros_like(cl_gg)
            cl_gi = np.zeros_like(cl_gg)

            delta_chi = quant['chi'][1:] - quant['chi'][:-1]

            for index_ell in range(self.settings.nellsmax):

                # find cl_integrand = (g(r) / r)**2 * P(l/r,z(r))
                for zbin1 in range(self.nzbins):
                    for zbin2 in range(zbin1 + 1):
                        cl_gg_integrand[1:, zbin1, zbin2] = kernel[1:, zbin1] * \
                            kernel[1:, zbin2] / quant['chi'][1:]**2 * pk_matter[index_ell, 1:]

                        # we replace syst['A_IA'] by 1.0 because we will model spectra separately
                        factor_ia = KiDSlike.get_factor_ia(quant, self.redshifts, 1.0)
                        cl_ii_integrand[1:, zbin1, zbin2] = pr_chi[1:, zbin1] * pr_chi[1:, zbin2] * \
                            factor_ia**2 / quant['chi'][1:]**2 * pk_matter[index_ell, 1:]

                        fact = (kernel[1:, zbin1] * pr_chi[1:, zbin2] + kernel[1:, zbin2] *
                                pr_chi[1:, zbin1]) * factor_ia / quant['chi'][1:]**2
                        cl_gi_integrand[1:, zbin1, zbin2] = fact * pk_matter[index_ell, 1:]

                for zbin1 in range(self.nzbins):
                    for zbin2 in range(zbin1 + 1):
                        cl_gg[index_ell, zbin1, zbin2] = np.sum(
                            0.5 * (cl_gg_integrand[1:, zbin1, zbin2] + cl_gg_integrand[:-1, zbin1, zbin2]) * delta_chi)

                        # here we divide by 16, because we get a 2^2 from g(z)!
                        cl_gg[index_ell, zbin1, zbin2] *= 9. / 16. * quant['omega_m']**2  # in units of Mpc**4
                        cl_gg[index_ell, zbin1, zbin2] *= (quant['small_h'] / 2997.9)**4  # dimensionless

                        cl_ii[index_ell, zbin1, zbin2] = np.sum(
                            0.5 * (cl_ii_integrand[1:, zbin1, zbin2] + cl_ii_integrand[:-1, zbin1, zbin2]) * delta_chi)
                        cl_gi[index_ell, zbin1, zbin2] = np.sum(
                            0.5 * (cl_gi_integrand[1:, zbin1, zbin2] + cl_gi_integrand[:-1, zbin1, zbin2]) * delta_chi)

                        # here we divide by 4, because we get a 2 from g(r)!
                        cl_gi[index_ell, zbin1, zbin2] *= 3. / 4. * quant['omega_m']
                        cl_gi[index_ell, zbin1, zbin2] *= (quant['small_h'] / 2997.9)**2

            # ordering of redshift bins is correct in definition of theory below!
            theory_ee_gg = np.zeros((self.nzcorrs, self.band_offset_ee), 'float64')
            theory_ee_ii = np.zeros((self.nzcorrs, self.band_offset_ee), 'float64')
            theory_ee_gi = np.zeros((self.nzcorrs, self.band_offset_ee), 'float64')

            index_corr = 0

            # perform interpolation
            for zbin1 in range(self.nzbins):
                for zbin2 in range(zbin1 + 1):

                    cl_sample_gg = cl_gg[:, zbin1, zbin2]
                    spline_cl_gg = itp.splrep(self.ells, cl_sample_gg)
                    d_l_ee_gg = self.ell_norm * itp.splev(self.ells_sum, spline_cl_gg)
                    theory_ee_gg[index_corr, :] = self.get_theory(
                        self.ells_sum, d_l_ee_gg, self.band_window_matrix, index_corr, band_type_is_ee=True)

                    cl_sample_gi = cl_gi[:, zbin1, zbin2]
                    spline_cl_gi = itp.splrep(self.ells, cl_sample_gi)
                    d_l_ee_gi = self.ell_norm * itp.splev(self.ells_sum, spline_cl_gi)
                    theory_ee_gi[index_corr, :] = self.get_theory(
                        self.ells_sum, d_l_ee_gi, self.band_window_matrix, index_corr, band_type_is_ee=True)

                    cl_sample_ii = cl_ii[:, zbin1, zbin2]
                    spline_cl_ii = itp.splrep(self.ells, cl_sample_ii)
                    d_l_ee_ii = self.ell_norm * itp.splev(self.ells_sum, spline_cl_ii)
                    theory_ee_ii[index_corr, :] = self.get_theory(
                        self.ells_sum, d_l_ee_ii, self.band_window_matrix, index_corr, band_type_is_ee=True)

                    index_corr += 1

            ee_bp = theory_ee_gg.flatten()
            gi_bp = theory_ee_gi.flatten()
            ii_bp = theory_ee_ii.flatten()

            return ee_bp, gi_bp, ii_bp

    def systematics_calc(self, par):
        '''
        Inputs:
            par (np.ndarray): array of parameters

        Outputs:
            bb_bp (np.ndarray): BB band powers

            ee_bp_noisy (np.ndarray): noise contribution to BB band powers

            bb_bp_noisy (np.ndarray): noise contrubution to EE band powers
        '''

        # dictionary for all parameters
        cosmo_par, syst, other, neutrino = KiDSlike.dictionary_params(par)
        del cosmo_par, other, neutrino

        # record noise
        a_noise = np.zeros(self.nzbins)
        add_noise_power = np.zeros(self.nzbins, dtype=bool)

        for zbin in range(self.nzbins):
            param_name = 'A_noise_z{:}'.format(zbin + 1)

            if param_name in self.settings.use_nuisance:
                a_noise[zbin] = syst['A_n' + str(zbin + 1)]
                add_noise_power[zbin] = True

        # empty arrays for recording noise
        theory_bb = np.zeros((self.nzcorrs, self.band_offset_bb), 'float64')
        theory_noise_ee = np.zeros((self.nzcorrs, self.band_offset_ee), 'float64')
        theory_noise_bb = np.zeros((self.nzcorrs, self.band_offset_bb), 'float64')

        index_corr = 0

        for zbin1 in range(self.nzbins):
            for zbin2 in range(zbin1 + 1):

                if zbin1 == zbin2:
                    a_noise_corr = a_noise[zbin1] * self.sigma_e[zbin1]**2 / self.n_eff[zbin1]
                else:
                    a_noise_corr = 0.

                d_l_noise = self.ell_norm * a_noise_corr

                # because resetting bias is set to False
                theory_bb[index_corr, :] = 0.

                if add_noise_power.all():
                    theory_noise_ee[index_corr, :] = self.get_theory(
                        self.ells_sum, d_l_noise, self.band_window_matrix, index_corr, band_type_is_ee=True)
                    theory_noise_bb[index_corr, :] = self.get_theory(
                        self.ells_sum, d_l_noise, self.band_window_matrix, index_corr, band_type_is_ee=False)

                index_corr += 1

        # flatten all arrays
        bb_bp = theory_bb.flatten()
        ee_bp_noisy = theory_noise_ee.flatten()
        bb_bp_noisy = theory_noise_bb.flatten()

        return bb_bp, ee_bp_noisy, bb_bp_noisy

    def theory(self, par):
        '''
        Inputs
            par (np.ndarray): array for the parameters

        Output:
            theory_pred (np.ndarray): array for the predicted band powers
        '''

        # dictionary for all parameters
        cosmo_par, syst, other, neutrino = KiDSlike.dictionary_params(par)
        del cosmo_par, other, neutrino

        # calculate all the power spectra
        ee_band, gi_band, ii_band = self.lensing_bandpowers(par)

        # calculate the systematics
        bb_band, ee_noise, bb_noise = self.systematics_calc(par)

        # total lensing
        ee_tot = ee_band + syst['A_IA'] * gi_band + syst['A_IA']**2 * ii_band

        # vector
        ee_bb_vec = np.concatenate((ee_tot, bb_band))
        ee_bb_vec_noise = np.concatenate((ee_noise, bb_noise))

        # final prediction
        ee_bb_final = ee_bb_vec[self.indices_for_bands_to_use] + ee_bb_vec_noise[self.indices_for_bands_to_use]

        return ee_bb_final

    def loglikelihood(self, par):
        '''
        Inputs:
            par (np.ndarray): array of the parameters

            prior_dict (dict): dictionary for the prior

        Outputs:
            logl (float): the log-likelihood
        '''
        # check if parameters lie outside the box
        pri = [self.all_priors[i].pdf(par[i]) for i in range(len(par))]
        prodpri = np.prod(pri)

        if prodpri == 0.0:
            chi2 = 2e12

        else:
            # calculate the theory
            expectation = self.theory(par)

            # calculate the difference
            diff = self.band_powers - expectation

            # compute chi2
            y_dummy = solve_triangular(self.chol_factor, diff, lower=True)
            chi2 = y_dummy.dot(y_dummy)

        return -0.5 * chi2

    def read_redshifts(self):
        '''
        inputs:
            None

        outputs:
            z (np.ndarray): redshifts

            N(z) distribution (np.ndarray): (n x 3) array

            Area under the histogram (np.ndarray): (1 x 3) array

            zmax (float): maximum redshift in the sample
        '''

        # the redshift distribution
        # +1 because we start from z = 0
        pr_red = np.zeros((self.settings.nzmax + 1, self.nzbins))

        # normalised redshift distribution
        pr_red_norm = np.zeros(self.nzbins, 'float64')

        if self.settings.bootstrap_photoz_errors:
            random_index_bootstrap = np.random.randint(
                int(self.settings.index_bootstrap_low), int(self.settings.index_bootstrap_high) + 1)

        for zbin in range(self.nzbins):
            redshift_bin = self.redshift_bins[zbin]

            if not self.settings.bootstrap_photoz_errors:
                window_file_path = os.path.join(self.settings.data_directory,
                                                '{:}/n_z_avg_{:}.hist'.format(self.settings.photoz_method, redshift_bin))
            else:
                window_file_path = os.path.join(
                    self.settings.data_directory,
                    '{:}/bootstraps/{:}/n_z_avg_bootstrap{:}.hist'.format(
                        self.settings.photoz_method,
                        redshift_bin,
                        random_index_bootstrap))

            zptemp, hist_pz = np.loadtxt(window_file_path, usecols=[0, 1], unpack=True)
            shift_to_midpoint = np.diff(zptemp)[0] / 2.

            z_samples = np.concatenate((np.zeros(1), zptemp + shift_to_midpoint))
            hist_samples = np.concatenate((np.zeros(1), hist_pz))

            # we assume that the histograms loaded are given as left-border histograms
            # and that the z-spacing is the same for each histogram

            # Spline
            spline_pz = itp.splrep(z_samples, hist_samples)

            redshifts = z_samples[0:]
            mask_min = redshifts >= z_samples.min()
            mask_max = redshifts <= z_samples.max()
            mask = mask_min & mask_max

            # points outside the z-range of the histograms are set to 0!
            pr_red[mask, zbin] = itp.splev(redshifts[mask], spline_pz)

            # Normalize selection functions
            delta_z = redshifts[1:] - redshifts[:-1]
            pr_red_norm[zbin] = np.sum(0.5 * (pr_red[1:, zbin] + pr_red[:-1, zbin]) * delta_z)

        zmax = np.max(redshifts)

        return redshifts, pr_red, pr_red_norm, zmax

    def bandpowers_and_cov(self):
        '''
        Calculate the data and the covariance matrix, given the m_correction

        We also store the cholesky factor so that we don't invert the covariance matrix
        in each likelihood computation

        Inputs:
            None

        Outputs:
            bandpowers (np.ndarray): the new bandpowers

            covariance (np.ndarray): the new covariance matrix
        '''

        # element-wise division of the covariance matrix
        cov_new = self.covariance / np.asarray(self.m_cov)

        # choose which elements are going to be used
        cov_new = self.covariance[np.ix_(self.indices_for_bands_to_use, self.indices_for_bands_to_use)]

        # element-wise division of the band powers
        bp_new = self.band_powers / np.asarray(self.m_vec)

        # choose which band powers are to be used
        bp_new = self.band_powers[self.indices_for_bands_to_use]

        # computes the cholesky factor
        chol_factor = cholesky(cov_new, lower=True)

        return bp_new, cov_new, chol_factor

    def calc_m_correction(self):
        '''
        Calculates the m-correction vector and matrix

        Inputs:
            None

        Outputs:
            m-correction vector, m-correction matrix

        Because we are using MOPED, the m-correction is fixed to the fiducial value.
        '''

        m_corr_per_zbin = self.m_corr_fiducial_per_zbin

        index_corr = 0

        for zbin1 in range(self.nzbins):
            for zbin2 in range(zbin1 + 1):

                # calculate m-correction vector here:
                # this loop goes over bands per z-corr
                # m-correction is the same for all bands in one tomographic bin.

                val_m_corr_ee = (1. + m_corr_per_zbin[zbin1]) * (1. +
                                                                 m_corr_per_zbin[zbin2]) * np.ones(len(self.settings.bands_EE_to_use))
                val_m_corr_bb = (1. + m_corr_per_zbin[zbin1]) * (1. +
                                                                 m_corr_per_zbin[zbin2]) * np.ones(len(self.settings.bands_BB_to_use))

                if index_corr == 0:
                    m_corr_ee = val_m_corr_ee
                    m_corr_bb = val_m_corr_bb
                else:
                    m_corr_ee = np.concatenate((m_corr_ee, val_m_corr_ee))
                    m_corr_bb = np.concatenate((m_corr_bb, val_m_corr_bb))

                index_corr += 1

        m_corr = np.concatenate((m_corr_ee, m_corr_bb))

        # this is required for scaling of covariance matrix:
        m_corr_matrix = np.matrix(m_corr).T * np.matrix(m_corr)

        return m_corr, m_corr_matrix

    def baryon_feedback_bias_sqr(self, k, redshift, a_bary=1.):
        """
        Fitting formula for baryon feedback following equation 10 and Table 2 from J. Harnois-Deraps et al. 2014 (arXiv.1407.4301)

        Inputs:
            k (np.ndarray): the wavevector

            z (np.ndarray): the redshift

            A_bary (float): the free amplitude for baryon feedback

        Outputs:

            b^2(k,z): bias squared
        """

        baryon_model = self.settings.baryon_model

        # k is expected in h/Mpc and is divided in log by this unit...
        x_wav = np.log10(k)

        # calculate a
        a_factor = 1. / (1. + redshift)

        # a squared
        a_sqr = a_factor * a_factor

        constant = {'AGN': {'A2': -0.11900, 'B2': 0.1300, 'C2': 0.6000, 'D2': 0.002110, 'E2': -2.0600,
                            'A1': 0.30800, 'B1': -0.6600, 'C1': -0.7600, 'D1': -0.002950, 'E1': 1.8400,
                            'A0': 0.15000, 'B0': 1.2200, 'C0': 1.3800, 'D0': 0.001300, 'E0': 3.5700},
                    'REF': {'A2': -0.05880, 'B2': -0.2510, 'C2': -0.9340, 'D2': -0.004540, 'E2': 0.8580,
                            'A1': 0.07280, 'B1': 0.0381, 'C1': 1.0600, 'D1': 0.006520, 'E1': -1.7900,
                            'A0': 0.00972, 'B0': 1.1200, 'C0': 0.7500, 'D0': -0.000196, 'E0': 4.5400},
                    'DBLIM': {'A2': -0.29500, 'B2': -0.9890, 'C2': -0.0143, 'D2': 0.001990, 'E2': -0.8250,
                              'A1': 0.49000, 'B1': 0.6420, 'C1': -0.0594, 'D1': -0.002350, 'E1': -0.0611,
                              'A0': -0.01660, 'B0': 1.0500, 'C0': 1.3000, 'D0': 0.001200, 'E0': 4.4800}}

        a_z = constant[baryon_model]['A2'] * a_sqr + \
            constant[baryon_model]['A1'] * a_factor + constant[baryon_model]['A0']
        b_z = constant[baryon_model]['B2'] * a_sqr + \
            constant[baryon_model]['B1'] * a_factor + constant[baryon_model]['B0']
        c_z = constant[baryon_model]['C2'] * a_sqr + \
            constant[baryon_model]['C1'] * a_factor + constant[baryon_model]['C0']
        d_z = constant[baryon_model]['D2'] * a_sqr + \
            constant[baryon_model]['D1'] * a_factor + constant[baryon_model]['D0']
        e_z = constant[baryon_model]['E2'] * a_sqr + \
            constant[baryon_model]['E1'] * a_factor + constant[baryon_model]['E0']

        # original formula:
        # bias_sqr = 1.-A_z*np.exp((B_z-C_z)**3)+D_z*x*np.exp(E_z*x)
        # original formula with a free amplitude A_bary:
        bias_sqr = 1. - a_bary * (a_z * np.exp((b_z * x_wav - c_z)**3) - d_z * x_wav * np.exp(e_z * x_wav))

        return bias_sqr

    @staticmethod
    def get_factor_ia(dict_quant, redshift, amplitude, exponent=0.0):
        '''
        Inputs:
            redshift (np.ndarray): the redshifts in the inference engine

            linear_growth_rate (np.ndarray): linear growth rate computed from clASS

            amplitude (float): the (free) amplitude for intrinsic alignment

            exponent (float): default (zero) if intrinsic alignment is True
        '''

        # in Mpc^3 / M_sol
        const = 5e-14 / dict_quant['small_h']**2

        # arbitrary convention
        redshift_0 = 0.3
        factor = -1. * amplitude * const * dict_quant['rho_crit'] * dict_quant['omega_m'] / \
            dict_quant['lgr'][1:] * ((1. + redshift[1:]) / (1. + redshift_0))**exponent

        return factor

    @staticmethod
    def get_critical_density(small_h):
        """
        The critical density of the Universe at redshift 0.

        Returns
        -------
        rho_crit in solar masses per cubic Megaparsec.

        """

        # Some constants
        mpc_cm = 3.08568025e24  # cm
        mass_sun_g = 1.98892e33  # g
        grav_const_mpc_mass_sun_s = mass_sun_g * (6.673e-8) / mpc_cm**3.
        h_100_s = 100. / (mpc_cm * 1.0e-5)  # s^-1

        rho_crit_0 = 3. * (small_h * h_100_s)**2. / (8. * np.pi * grav_const_mpc_mass_sun_s)

        return rho_crit_0

    def get_theory(self, ells_sum, d_l, band_window_matrix, index_corr, band_type_is_ee=True):
        '''
        Compression 1: Power spectra are converted into band powers

        Inputs:
            ells_sum (np.ndarray): these are the ell modes

            d_l (np.ndarray): these are the interpolated power spectra

            band_window_matrix (np.ndarray): the band window matrix provided by KiDS-450

            index_corr (int): the position in the auto- and cross-power, for example, 0: 00, 1: 01

            band_type_is_ee (bool): True if we are working with ee band powers

        Outputs:
            Band powers
        '''

        # these slice out the full ee --> ee and BB --> BB block of the full BWM!
        # sp: slicing points
        sp_ee_x = (0, self.nzcorrs * self.band_offset_ee)
        sp_ee_y = (0, self.nzcorrs * len(self.ells_intp))
        sp_bb_x = (self.nzcorrs * self.band_offset_ee, self.nzcorrs * (self.band_offset_bb + self.band_offset_ee))
        sp_bb_y = (self.nzcorrs * len(self.ells_intp), 2 * self.nzcorrs * len(self.ells_intp))

        if band_type_is_ee:
            sp_x = sp_ee_x
            sp_y = sp_ee_y
            band_offset = self.band_offset_ee
        else:
            sp_x = sp_bb_x
            sp_y = sp_bb_y
            band_offset = self.band_offset_bb

        bwm_sliced = band_window_matrix[sp_x[0]:sp_x[1], sp_y[0]:sp_y[1]]

        bands = range(index_corr * band_offset, (index_corr + 1) * band_offset)

        d_avg = np.zeros(len(bands))

        for index_band, alpha in enumerate(bands):
            # jump along tomographic auto-correlations only:
            index_ell_low = int(index_corr * len(self.ells_intp))
            index_ell_high = int((index_corr + 1) * len(self.ells_intp))
            spline_w_alpha_l = itp.splrep(self.ells_intp, bwm_sliced[alpha, index_ell_low:index_ell_high])
            d_avg[index_band] = np.sum(itp.splev(ells_sum, spline_w_alpha_l) * d_l)

        return d_avg

if __name__ == '__main__':

    kids = KiDSlike(file_settings = '../settings')
    start_point = np.array([0.1295, 0.0224, 2.895, 0.9948, 0.7411, 1.0078, 0.5692, 0.0289, 0.0133, -0.0087, -1.9163])
    kids.lensing_new(start_point)
    kids.lensing_bandpowers(start_point)
