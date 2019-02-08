# -*- coding: utf-8 -*-
"""
Created on Fri Jan 18 14:10:58 2019

@author: DEPAGRA
"""
import libdtw as lib
import sys
import numpy as np
import pickle

np.set_printoptions(precision=2)

try:
    N_DATA = int(sys.argv[1])
except LookupError as ex:
    N_DATA = 31

DATA = lib.load_data(N_DATA)
D = lib.Dtw(DATA)

step_pattern = 'symmetric2'
file_path = 'dtwObjOptWeights%dAllFeats.pickle' % N_DATA
try:
    with open(file_path, 'rb') as f:
        D.data['feat_weights'] = pickle.load(f)  
    print(D.data['feat_weights'])
except OSError as ex:
    pass
finally:
    D.optimize_weigths(step_pattern, n_steps = 5, file_path = file_path, n_jobs=7)
    with open(file_path, 'wb') as f:
        pickle.dump(D.data['feat_weights'], f, protocol=pickle.HIGHEST_PROTOCOL)

D.plot_weights()
