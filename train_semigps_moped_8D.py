'''
Author: Arrykrishna Mootoovaloo
Collaborators: Alan Heavens, Andrew Jaffe, Florent Leclercq
Email : a.mootoovaloo17@imperial.ac.uk
Affiliation : Imperial Centre for Inference and Cosmology
Status : Under Development
Description : Script to train all Gaussian Processes
'''

import numpy as np
import emulator.training as tr
import emulator.utils.helpers as hp

# specify where we want to save the GPs (not Dropbox!)
SAVEDIR = '/home/harry/Desktop/kids-paper-2/dev/'

# LHS method
lhs_method = 'maximin_3000'

def run_optimisation(spectrum = 'total_', ndim=8):
    '''
    Train GPs related to each compressed data

    Inputs
    ------
    spectrum (str) : gg, gi or ii

    ndim (int) : number of input dimensions

    Returns
    -------
    '''

    # load the training set
    spec = hp.load_arrays('simulations_mean', spectrum)

    # inputs to the GP
    inputs = spec[:, 0:ndim]

    # number of MOPED coefficients
    n_moped = spec.shape[1] - ndim

    for i in range(n_moped):

        # choose the output
        y_output = spec[:, ndim + i]

        # train the GP
        trained_gp = tr.training(file_settings='settings', x_train=inputs, y_train=y_output)

        # save the GP
        hp.store_pkl_file(trained_gp, SAVEDIR + 'gps/' + spectrum, 'gp_' + str(i))

# train each compressed datum for each spectrum type 
run_optimisation('total_' + lhs_method, ndim=8)