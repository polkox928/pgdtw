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
    print(D.data['feat_weights'])
except OSError as ex:
    pass
finally:
    D.optimize_weights(step_pattern, n_steps = 10, file_path = file_path, n_jobs=n_jobs)
    with open(file_path, 'wb') as f:
        pickle.dump(D.data['feat_weights'], f, protocol=pickle.HIGHEST_PROTOCOL)


D.plot_weights(n=len(D.data['feat_weights']))

D.plot_by_name('5154', 'ba_TZWZNzFFdHb')
sum(D.data['feat_weights'])

#%%
feat_weight = sorted(list(D.get_weight_variables().items()), key=lambda x: x[1], reverse=True)

import matplotlib.pyplot as plt

for (name, weight), i in zip(feat_weight, range(1, len(feat_weight)+1)):
    plt.figure()
    D.plot_by_name('5153', name)
    plt.title(name + ' in Reference batch, weight: %0.3f'%weight)
    plt.xlabel('Time (min)')
    plt.ylabel('PV value')
    plt.savefig('pics/weights/ref_%02d_%s'%(i, name))
    plt.close()
    plt.figure()
    for _id in D.data['queriesID']:
        D.plot_by_name(_id, name)
        plt.title(name + ' in query batches')
        plt.xlabel('Time (min)')
    plt.savefig('pics/weights/queries_%02d_%s'%(i, name))
    plt.close()
        
        
    
    
#%%
for _id in D.data['queriesID']:
    D.plot_by_name(_id, feat_weight[0][0])
#%%
for _id in D.data['queriesID']:
    D.plot_by_name(_id, feat_weight[4][0])
