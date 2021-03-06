'''
Author: Arrykrishna Mootoovaloo
Collaborators: Alan Heavens, Andrew Jaffe, Florent Leclercq
Email : a.mootoovaloo17@imperial.ac.uk
Affiliation : Imperial Centre for Inference and Cosmology
Status : Under Development
Description : Script to train all Gaussian Processes (zero mean Gaussian Process)
'''

import numpy as np
import emulator.zerogp as gp
import emulator.utils.helpers as hp

# specify where we want to save the GPs (not Dropbox!)
SAVEDIR = '/home/harry/Desktop/kids-paper-2/dev/'

# LHS method
lhs_method = 'maximin_1000'

# settings for the Gaussian Process model
ndim = 7
bounds = np.repeat(np.array([[-1.5, 6]]), ndim + 1, axis=0)
bounds[0] = np.array([-1, 1])


def run_optimisation(spectrum, ndim=7):
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

        # the zero meanGP script takes both inputs and output together
        data_selected = np.concatenate((inputs, np.atleast_2d(y_output).T), axis=1)

        # train the GP
        trained_gp = gp.GAUSSIAN_PROCESS(data_selected, sigma=[-40], train=True, nrestart=5)
        trained_gp.transform()
        trained_gp.fit(method='L-BFGS-B', bounds=bounds, options={'ftol': 1E-12, 'maxiter': 500})

        # save the GP
        hp.store_pkl_file(trained_gp, SAVEDIR + 'gps/' + 'zero_' + spectrum, 'gp_' + str(i))

        print('Training GP for ' + spectrum + '_' + str(i) + ' completed')


# train each compressed datum for each spectrum type (GG, GI or II)
run_optimisation('gg_' + lhs_method, ndim=7)
run_optimisation('gi_' + lhs_method, ndim=7)
run_optimisation('ii_' + lhs_method, ndim=7)
