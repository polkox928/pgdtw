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
    N_DATA = 5

DATA = lib.load_data(N_DATA)
D = lib.Dtw(DATA)

step_pattern = 'symmetric2'

try:
    with open('dtwObjOptWeights%dAllFeats.pickle' % N_DATA, 'rb') as f:
        D.data['feat_weights'] = pickle.load(f)
except OSError as ex:
    D.optimize_weigths(step_pattern)
    with open('dtwObjOptWeights%dAllFeats.pickle' % N_DATA, 'wb') as f:
        pickle.dump(D.data['feat_weights'], f, protocol=pickle.HIGHEST_PROTOCOL)

# D.weight_optimization_single_batch(D.data['queriesID'][1], 'symmetricP2')


D.plot_weights()
