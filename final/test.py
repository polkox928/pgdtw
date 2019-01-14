#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 28 16:39:46 2018

@author: paolograniero
"""
import libdtw as lib
import numpy as np

ref = np.array([1,2,3,4,5,6]).reshape(-1,1)
query = np.array([5,6,7]).reshape(-1,1)

d = lib.dtw()

distMatrix = d.CompDistMatrix(ref, query, "euclidean", 1)

print(distMatrix)

acc = d.CompAccDistmatrix(distMatrix, "symmetricP05")

print(acc)
