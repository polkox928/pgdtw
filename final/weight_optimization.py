# -*- coding: utf-8 -*-
"""
Created on Fri Jan 18 14:10:58 2019

@author: DEPAGRA
"""

import libdtw as lib
import sys
import numpy as np
np.set_printoptions(precision=3)

try:
    N_DATA = int(sys.argv[1])
except LookupError as ex:
    N_DATA = 3
    
DATA = lib.load_data(N_DATA)
D = lib.Dtw(DATA)

step_pattern = 'symmetric2'

D.optimize_weigths(step_pattern)

print(D.data['feat_weights'])

#D.weight_optimization_single_batch(D.data['queriesID'][1], 'symmetricP2')
