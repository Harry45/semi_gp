'''
Author: Arrykrishna Mootoovaloo
Collaborators: Alan Heavens, Andrew Jaffe, Florent Leclercq
Email : a.mootoovaloo17@imperial.ac.uk
Affiliation : Imperial Centre for Inference and Cosmology
Status : Under Development
Description : Simulations for building an emulator for sigma_12

The following parameters and prior box are used (be careful about the order):

prior['omega_cdm'] = [0.010, 0.390, 'uniform']

prior['omega_b'] = [0.019, 0.007, 'uniform']

prior['ln10^{10}A_s'] = [1.700, 3.300, 'uniform']

prior['n_s'] = [0.700, 0.600, 'uniform']

prior['h'] = [0.640, 0.180, 'uniform']

prior['m_ncdm'] = [0.06, 0.94, 'uniform']
'''

import numpy as np
import pandas as pd
import emulator.kids_likelihood as kd
import emulator.utils.helpers as hp

# LHS method
lhs_method = 'maximin_1000'

# Radius in Mpc
radius = 8

# fixed redshift
redshift = 0.0

# load the LHS points
lhs_points = pd.read_csv('lhs/' + lhs_method + '_6D', index_col=0).values
n_train = lhs_points.shape[0]

# transform to the pre-defined prior box
min_max_prior = np.array([[0.010, 0.400],
                          [0.019, 0.026],
                          [1.700, 5.000],
                          [0.700, 1.300],
                          [0.640, 0.820],
                          [0.060, 1.00]])

scaled = min_max_prior[:, 0] + (min_max_prior[:, 1] - min_max_prior[:, 0]) * lhs_points

# calll the KiDs likelihood
kids = kd.KiDSlike(file_settings = 'settings')

# create an empty array for recording values of sigma
sigma = np.zeros(n_train)

for i in range(n_train):
	sigma[i] = kids.calculate_sigma(scaled[i], radius, redshift)

# build the training set
training_set = np.concatenate((scaled, np.atleast_2d(sigma).T), axis = 1)

# save the traiing set
hp.store_arrays(training_set, 'simulations_sigma', 'sigma_eight_' + lhs_method)