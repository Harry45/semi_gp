'''
Author: Arrykrishna Mootoovaloo
Collaborators: Alan Heavens, Andrew Jaffe, Florent Leclercq
Email : a.mootoovaloo17@imperial.ac.uk
Affiliation : Imperial Centre for Inference and Cosmology
Status : Under Development
Description : Main Script for running MCMC
'''
import numpy as np
from emulator.mcmc import MCMC

MCMC_ROUTINE = MCMC(settings='settings', compression=True)
start_point = np.array([0.1295, 0.0224, 2.895, 0.9948, 0.7411, 1.0078, 0.5692, 0.0289, 0.0133, -0.0087, -1.9163])

# this is done once
# B, y = MCMC_ROUTINE.moped.compression(eps = 1E-6, parameters = start_point)
# MCMC_ROUTINE.moped.save_vectors('moped/')

# load the compressed data
B, y = MCMC_ROUTINE.moped.load_vectors('moped/')

# load all the GPs
# gps = MCMC_ROUTINE.moped.load_gps('GPS')

# Experiments to perform

# compression = True, bootstrap_photoz_errors = False, emulator = False
# class_moped_mean_nz

# compression = True, bootstrap_photoz_errors = True, emulator = False
# class_moped_samples_nz

# compression = False, bootstrap_photoz_errors = False, emulator = False
# class_bandpowers_mean_nz

# compression = False, bootstrap_photoz_errors = True, emulator = False
# class_bandpowers_samples_nz

# compression = True, bootstrap_photoz_errors = False, emulator = True
# emulator_moped_mean_nz

# compression = True, bootstrap_photoz_errors = True, emulator = True
# emulator_moped_samples_nz

print(MCMC_ROUTINE.loglike(start_point))
sampler = MCMC_ROUTINE.posterior_sampling(start_point, 'class_moped_mean_nz')