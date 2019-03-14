# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 15:10:23 2019

@author: DEPAGRA
"""
#%%
import matplotlib.pyplot as plt
import numpy as np

n_rows = 500
n_cols = 400
X = np.ones((n_rows, n_cols, 3), dtype = np.int64)*255

def itakura(p, i, j, ref_len, query_len):
    in_domain = (i >= np.floor(j*p/(p+1))) and \
                (i <= np.ceil(j*(p+1)/p)) and \
                (i <= np.ceil(ref_len+(j-query_len)*(p/(p+1)))) and \
                (i >= np.floor(ref_len+(j-query_len)*((p+1)/p)))
    return in_domain

p = 0.5

for i in range(n_rows):
    for j in range(n_cols):
        if itakura(p, i, j, n_rows, n_cols):
            X[i, j, 0] = 210
            X[i, j, 1] = 210
            X[i, j, 2] = 210
            
plt.imshow(X)
plt.xlabel('Query series index')
plt.ylabel('Reference series index')
plt.gca().invert_yaxis()
plt.title('Itakura Parallelogram p = 0.5')
plt.savefig('images_thesis/itakura.png')