'''
Author: Arrykrishna Mootoovaloo
Collaborators: Alan Heavens, Andrew Jaffe, Florent Leclercq
Email : a.mootoovaloo17@imperial.ac.uk
Affiliation : Imperial Centre for Inference and Cosmology
Status : Under Development
Description : Simulations for building the emulator

Note that we are building an 8-dimensional emulator based on 8 inputs, that is, we are including the intrinsic alignment parameter in the compressed data.
'''

import numpy as np
import pandas as pd
import emulator.moped as emu_moped
import emulator.utils.helpers as hp

# LHS method
lhs_method = 'maximin_3000'

# load the LHS points
lhs_points = pd.read_csv('lhs/' + lhs_method + '_8D', index_col=0).values
n_train = lhs_points.shape[0]

# transform to the pre-defined prior box
min_max_prior = np.array([[0.010, 0.400],
                          [0.019, 0.026],
                          [1.700, 5.000],
                          [0.700, 1.300],
                          [0.640, 0.820],
                          [0.000, 2.0],
                          [0.060, 1.00],
                          [-6.00, 6.00]])

scaled = min_max_prior[:, 0] + (min_max_prior[:, 1] - min_max_prior[:, 0]) * lhs_points

# the cosmology part is independent of the nuisance parameters
# so we concatenate a matrix of zeros
# 4 because we have 4 additional nuisance parameters
# but the last column must contain the parameter A_IA
# the parameters are aligned as follows

# --------------------------------------------------
# prior['omega_cdm'] = [0.010, 0.390, 'uniform']
# prior['omega_b'] = [0.019, 0.007, 'uniform']
# prior['ln10^{10}A_s'] = [1.700, 3.300, 'uniform']
# prior['n_s'] = [0.700, 0.600, 'uniform']
# prior['h'] = [0.640, 0.180, 'uniform']
# prior['A_bary'] = [0.000, 2.00, 'uniform']
# prior['m_ncdm'] = [0.06, 0.94, 'uniform']
# --------------------------------------------------
# prior['A_n1'] = [-0.100, 0.200, 'uniform']
# prior['A_n2'] = [-0.100, 0.200, 'uniform']
# prior['A_n3'] = [-0.100, 0.200, 'uniform']
# --------------------------------------------------
# prior['A_IA'] = [-6.00, 12.00, 'uniform']
# --------------------------------------------------

training_points = np.concatenate((scaled[:, 0:-1], np.zeros((n_train, 3)), scaled[:, -1].reshape(3000, 1)), axis=1)

# load the MOPED module to run the simulations at these points
moped = emu_moped.MOPED('settings')
B, y = moped.load_vectors('moped')

# number of MOPED coefficients
n_moped = len(y)

# create empty arrays to store the MOPED coefficients
record_total = np.zeros((n_train, n_moped))

# run the algorithm at these points
for i in range(n_train):
    record_total[i] = moped.compress_theory_total(training_points[i])

# store the inputs and outputs to the emulator
training_points_total = np.concatenate((scaled, record_total), axis=1)

# drop rows which contain NaNs if any
training_points_total = training_points_total[~np.isnan(training_points_total).any(axis=1)]

# save the training points
hp.store_arrays(training_points_total, 'simulations_mean', 'total_' + lhs_method)
