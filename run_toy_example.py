#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), \
                                             "../src")))
import time
import numpy as np
import pickle
from EM_learning import hmc_var_parameter_learning



#-----------data initialization Simulated PHMC-VAR
#hyper-parameters
X_dim = 2
X_order = 1
nb_regimes = 3
innovation = "gaussian"


data_file = "toy_dataset.pkl"

#----training data loading
infile = open(data_file, 'rb')
data_set = pickle.load(infile)
infile.close()

#time series and associated initial values
data = data_set[1]
initial_values = data_set[2]

#----toy-dataset visualization
print("--------Toy dataset visualization")
print("Nb time series = ", len(data))
print("Displays frist timeseries : ")
print("Initial_values = ", initial_values[0][0])
print("The remaining observations = ", data[0][0])
print("-----------------------------------")


#----Unsupervised learning scheme: all states are possible at each time-steps
# list of length S
states = []  
S = len(data)
for s in range(S):
    
    #list of T_s arrays
    states.append([])
    T_s = data[s].shape[0]   
    for t in range(T_s):
        states[s].append( np.array([i for i in range(nb_regimes)], \
                                                              dtype=np.int32) )
    
#----model learning

#running time estimation starts
start_time = time.time()

(total_log_ll, A, Pi, list_Gamma, list_Alpha, ar_coefficients, sigma, \
 intercept, psi) = hmc_var_parameter_learning (X_dim, X_order, nb_regimes, \
                                               data, initial_values, states, \
                                               innovation, \
                                               var_init_method='rand1', \
                                               hmc_init_method='rand', \
                                               nb_iters=100, epsilon=1e-6, \
                                               nb_init=10, nb_iters_init=5)

#running time estimation ends
duration = time.time() - start_time
print("-----------------------------------------------------------")
print("Learning algorithm lastes {} minutes".format(duration/60))
print("-----------------------------------------------------------")

#vizualize some parameters
print("==============================OUTPUT==================================")
print("------------------AR process----------------------")
print("total_log_ll= ", total_log_ll)
print("ar_coefficients=", ar_coefficients)
print("intercept=", intercept)
print("sigma=", sigma)
print("psi=", psi)
print()
print("------------------Markov chain----------------------")
print("Pi=", Pi)
print("A=", A)


