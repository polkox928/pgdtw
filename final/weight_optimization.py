# -*- coding: utf-8 -*-
"""
Created on Fri Jan 18 14:10:58 2019

@author: DEPAGRA
"""

import libdtw as lib
import sys
import numpy as np
import pickle
from pprint import pprint
import matplotlib.pyplot as plt

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

# D.weight_optimization_single_batch(D.data['queriesID'][1], 'symmetricP2')
with open('dtwObjOptWeights%dAllFeats.pickle' % N_DATA, 'wb') as f:
    pickle.dump(D.data['feat_weights'], f, protocol=pickle.HIGHEST_PROTOCOL)

plt.rcdefaults()
fig, ax = plt.subplots(figsize=(15, 8))

vars = sorted(list(D.get_weight_variables().items()), key=lambda x: x[1], reverse=True)[:25]
names = [v[0] for v in vars]
y_pos = np.arange(len(names))
weights = [v[1] for v in vars]

ax.barh(y_pos, weights, align='center', color='#d90000')
ax.set_yticks(y_pos)
ax.set_yticklabels(names)
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('Weights')
ax.set_title('Variables\' weights')

fig.tight_layout()
plt.show()
