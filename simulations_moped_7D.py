'''
Author: Arrykrishna Mootoovaloo
Collaborators: Alan Heavens, Andrew Jaffe, Florent Leclercq
Email : a.mootoovaloo17@imperial.ac.uk
Affiliation : Imperial Centre for Inference and Cosmology
Status : Under Development
Description : Simulations for building the emulator
'''

import numpy as np
import pandas as pd
import emulator.moped as emu_moped
import emulator.utils.helpers as hp

# LHS method
lhs_method = 'random_1000'

# load the LHS points
lhs_points = pd.read_csv('lhs/' + lhs_method + '_7D', index_col=0).values
n_train = lhs_points.shape[0]

# transform to the pre-defined prior box
min_max_prior = np.array([[0.010, 0.400],
                          [0.019, 0.026],
                          [1.700, 5.000],
                          [0.700, 1.300],
                          [0.640, 0.820],
                          [0.000, 2.0],
                          [0.060, 1.00]])

scaled = min_max_prior[:, 0] + (min_max_prior[:, 1] - min_max_prior[:, 0]) * lhs_points

# the cosmology part is independent of the nuisance parameters
# so we concatenate a matrix of zeros
# 4 because we have 4 additional nuisance parameters
training_points = np.concatenate((scaled, np.zeros((n_train, 4))), axis=1)

# load the MOPED module to run the simulations at these points
moped = emu_moped.MOPED('settings')
B, y = moped.load_vectors('moped')

# number of MOPED coefficients
n_moped = len(y)

# create empty arrays to store the MOPED coefficients
record_gg = np.zeros((n_train, n_moped))
record_gi = np.zeros((n_train, n_moped))
record_ii = np.zeros((n_train, n_moped))

# run the algorithm at these points
for i in range(n_train):
    gg, gi, ii = moped.compress_theory_lensing(training_points[i])
    record_gg[i] = gg
    record_gi[i] = gi
    record_ii[i] = ii

# store the inputs and outputs to the emulator
training_points_gg = np.concatenate((scaled, record_gg), axis=1)
training_points_gi = np.concatenate((scaled, record_gi), axis=1)
training_points_ii = np.concatenate((scaled, record_ii), axis=1)

# drop rows which contain NaNs if any
training_points_gg = training_points_gg[~np.isnan(training_points_gg).any(axis=1)]
training_points_gi = training_points_gi[~np.isnan(training_points_gi).any(axis=1)]
training_points_ii = training_points_ii[~np.isnan(training_points_ii).any(axis=1)]

# save the training points
hp.store_arrays(training_points_gg, 'simulations_mean', 'gg_' + lhs_method)
hp.store_arrays(training_points_gi, 'simulations_mean', 'gi_' + lhs_method)
hp.store_arrays(training_points_ii, 'simulations_mean', 'ii_' + lhs_method)
