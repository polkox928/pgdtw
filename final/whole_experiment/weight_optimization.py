# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 10:06:01 2019

@author: DEPAGRA
"""

import libdtw as lib
import sys
import numpy as np
import pickle

np.set_printoptions(precision=2)

try:
    N_DATA = int(sys.argv[1]) if int(sys.argv[1]) >= 2 else 1000
except LookupError as ex:
    N_DATA = 1000

try:
    n_jobs = int(sys.argv[2])
except LookupError as ex:
    n_jobs = 1


DATA = lib.load_data(N_DATA)
D = lib.Dtw(DATA)

num_queries = D.data['num_queries']
step_pattern = 'symmetric2'
file_path = 'optWeights.pickle'

try:
    with open(file_path, 'rb') as f:
        D.data['feat_weights'] = pickle.load(f)
    print('Initial weights:\n', D.data['feat_weights'])
except OSError as ex:
    pass
finally:
    D.optimize_weights(step_pattern, n_steps = 20, file_path = file_path, n_jobs=n_jobs)
    with open(file_path, 'wb') as f:
        pickle.dump(D.data['feat_weights'], f, protocol=pickle.HIGHEST_PROTOCOL)