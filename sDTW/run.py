# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 14:36:04 2018

@author: DEPAGRA
"""
#%%


from DTWpg import dtw
from time import time
import pickle

with open("batch_data.pickle", "rb") as infile:
    batchData = pickle.load(infile)
start = time()


res = dtw(jsonObj = batchData, 
          open_ended = True, 
          all_subseq = True, 
          only_distance = True, 
          dist_measure = "euclidean", 
          n_jobs = -2)

print("Total time elapsed: {:.0f} seconds".format(time() - start))
