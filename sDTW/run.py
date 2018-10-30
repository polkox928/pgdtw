# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 14:36:04 2018

@author: DEPAGRA
"""

from DTWpg import dtw
from time import time
import pickle
import pandas as pd
from time import time
import pickle

with open("C:\\Users\\DEPAGRA\\Documents\\GitHub\\pgdtw\\data\\batch_data.pickle", "rb") as infile:
    batchData = pickle.load(infile)
start = time()

IDs = [_id for _id in batchData.keys() if isinstance(_id, int)]
n_batches = len(IDs)
columns = ["refID", "queryID", "t_query", "t_ref", "DTW_dist", "T_ref", "T_query"]

i = 1
for ID in IDs:
    batchData["reference"] = ID
    print("Processing Reference Batch {0} - {1}/{2}".format(ID, i, n_batches))
    
    res = dtw(jsonObj = batchData, 
              open_ended = True, 
              all_subseq = True, 
              only_distance = True, 
              dist_measure = "euclidean", 
              n_jobs = 1)
    
    historicalDataset = pd.DataFrame(columns = columns)
    for queryID, output in res.output.items():
        
        historicalDataset = historicalDataset.append(pd.DataFrame(output, columns = columns), ignore_index = True)
    i += 1

    historicalDataset.to_csv("data\\historicalDataset{}.csv".format(ID), header = True, index = False)

    print("Time elapsed: {0:.0f}:{1:02.0f} minutes\n".format(np.floor((time() - start)/60), (time()-start)%60))

print("Total time elapsed: {0:.0f}:{1:02.0f} minutes\n".format(np.floor((time() - start)/60), (time()-start)%60))
