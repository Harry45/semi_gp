'''
Author: Arrykrishna Mootoovaloo
Collaborators: Alan Heavens, Andrew Jaffe, Florent Leclercq
Email : a.mootoovaloo17@imperial.ac.uk
Affiliation : Imperial Centre for Inference and Cosmology
Status : Under Development
Description : train GPs for sigma_x (x = 8 or x = 12)
'''

import os
import numpy as np
import emulator.zerogp as gp
import emulator.utils.helpers as hp

# load the table
table = hp.load_arrays('simulations_sigma', 'sigma_twelve_maximin_1000')

# GP file name
file_name = 'gp_twelve'

# noise term in log
sigma = [-40.0]

# do we want to train the GP?
train = True

# number of restart - to avoid minima
nrestart = 5

# number of dimensions
ndim = 6

# set a bound for the optimiser - prior-like
bounds = np.repeat(np.array([[-1.5, 6]]), ndim + 1, axis=0)
bounds[0] = np.array([-1, 1])

# train the GP
sigma_gp = gp.GAUSSIAN_PROCESS(table, sigma=sigma, train=train, nrestart=nrestart)
sigma_gp.transform()
sigma_gp.fit(method='L-BFGS-B', bounds=bounds, options={'ftol': 1E-12, 'maxiter': 500})

# save GP to file
hp.store_pkl_file(sigma_gp, 'gps_sigma', file_name)

# do a quick test - since it is 6D, we should expect very small variance
test_point = np.array([0.15, 0.022, 2.47, 1.13, 0.75, 0.5])
print(sigma_gp.prediction(test_point, returnvar=True))
