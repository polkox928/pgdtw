# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 14:29:53 2019

@author: DEPAGRA
"""
#%%
import matplotlib.pyplot as plt
import numpy as np

n_rows = 25
n_cols = 20
X = np.ones((n_rows, n_cols, 3), dtype = np.int64)*255

B = 5

for i in range(n_rows):
    for j in range(n_cols):
        if abs(i-j) <= B:
            X[i, j, 0] = 210
            X[i, j, 1] = 210
            X[i, j, 2] = 210
            
plt.imshow(X)
plt.xlabel('Query series index')
plt.ylabel('Reference series index')
plt.gca().invert_yaxis()
plt.title('Sakoe-Chiba band (Q = 5)')
plt.savefig('images_thesis/sakoe-chiba.png')