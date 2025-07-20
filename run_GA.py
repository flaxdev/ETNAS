# -*- coding: utf-8 -*-
"""
Created on Thu Dec 26 22:50:36 2024

@author: flavio
"""

import pandas as pd
import numpy as np
import scipy
from BGA import genetic_algorithm
from EDmodelling import evalModelAgainstCrush






# load the dataset

# DATA = pd.read_pickle("data/total_dataset.pkl")
DATA = pd.read_pickle("data/dataset_2011_2021_complete.pkl")

# missing data imputation    
DATA.fillna(-1, inplace=True)
    
X = DATA.drop(['Target'], axis=1) 
# X = DATA[['Tremor','EWIP','Strain']]

y =  DATA['Target']  

# define the total iterations
n_iter = 5
# bits per variable
n_bits = X.shape[1]
# define the population size
n_pop = 7
# crossover rate
r_cross = 0.9
# mutation rate
r_mut = 1.0 / float(n_bits)


# set the inital default chormosome
chromosome = np.zeros(n_bits) 
iok = np.nonzero(np.in1d(X.columns.values, ['Tremor','EWIP','HMM','Strain']))[0]
chromosome[iok] = 1

# hyper-parameters distributions
CV_param_dist = {'max_depth': scipy.stats.randint(3, 8), 
                  'n_estimators': scipy.stats.randint(5, 15),
                  'min_samples_leaf': scipy.stats.randint(1, 3),
                  'hysteresis': scipy.stats.randint(5, 150)}                  


# perform the genetic algorithm search
fmin = lambda chromosomefeatures: evalModelAgainstCrush(chromosomefeatures, X, y, CV_param_dist)
best, score, Table, population = genetic_algorithm(fmin, n_bits, n_iter, n_pop, r_cross, r_mut, \
                                       chromosome=chromosome)
