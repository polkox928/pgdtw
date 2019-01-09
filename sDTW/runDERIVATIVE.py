# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 15:37:51 2018

@author: DEPAGRA
"""
# DERIVATIVE DTW
from DTWpg import dtw
from time import time
import pickle
import pandas as pd
import numpy as np
import os
from sklearn.ensemble import GradientBoostingRegressor
from joblib import Parallel, delayed
import multiprocessing

# File containing the batch data
with open("C:\\Users\\DEPAGRA\\Documents\\GitHub\\pgdtw\\data\\batch_data.pickle", "rb") as infile:
    batchData = pickle.load(infile)

IDs = [_id for _id in batchData.keys() if isinstance(_id, int)]
n_batches = len(IDs)
columns = ["refID", "queryID", "t_query", "t_ref", "DTW_dist", "T_ref", "T_query"]

print("PERFORMING derivativeDTW VARIATION")
start = time()


def processBatch(ID):
    batchData["reference"] = ID
    res = dtw(jsonObj = batchData, 
              open_ended = True, 
              all_subseq = True, 
              only_distance = True, 
              dist_measure = "euclidean",
              scale = True,
              shapeDTW = True,
              shape_descriptor = "derivative",
              n_jobs = 1)
    
    res.ApplyDTW()
    
    historicalDataset = pd.DataFrame(columns = columns)
    for queryID, output in res.output.items():
        
        historicalDataset = historicalDataset.append(pd.DataFrame(output, columns = columns), ignore_index = True)


    historicalDataset.to_csv("dataDerivative\\historical\\historicalDataset{}.csv".format(ID), header = True, index = False)
    return

num_cores = multiprocessing.cpu_count()

Parallel(n_jobs=num_cores, verbose = 10)(delayed(processBatch)(ID) for ID in IDs)

complete = pd.DataFrame(columns = columns)

for filename in os.listdir("dataDerivative\\historical"):
    if filename.startswith("historicalDataset"):
        complete = complete.append(pd.read_csv("dataDerivative\\historical\\"+filename, header = 0), ignore_index = True)
    
complete.drop_duplicates(inplace = True)    
complete.to_csv("dataDerivative\\historical\\completeDataset.csv", header = True, index = False)

train_columns = ["t_query", "t_ref", "T_ref", "DTW_dist"]
target = ["T_query"]


print("PREDICTION")

def predictBatch(queryID):
    
    model = GradientBoostingRegressor()

    train_dataset = complete.loc[(complete.queryID != queryID) & (complete.refID != queryID), :]
    test_dataset = complete.loc[complete.queryID == queryID , :]
    
    X = train_dataset.loc[:, train_columns].values
    y = train_dataset.loc[:, target].values.reshape(-1,)
        
    model.fit(X,y)
# =============================================================================
#     filename = 'model.sav'
#     pickle.dump(model, open(filename, 'wb'))
# =============================================================================
    X_test = test_dataset.loc[:, train_columns].values
    
    test_dataset.loc[:, "predGB"] = model.predict(X_test)
    test_dataset.loc[:, "predAbs"] = test_dataset["T_ref"] - test_dataset["t_ref"] + test_dataset["t_query"] 
    test_dataset = test_dataset.astype({
                                      "refID" : int, 
                                      "queryID" : int,
                                      "t_query": int, 
                                      "t_ref": int, 
                                      "T_ref": int, 
                                      "DTW_dist": float, 
                                      "T_query": int,
                                      "predGB": float,
                                      "predAbs": int})

    test_dataset = test_dataset.groupby(["t_query"]).aggregate({"predGB" : "mean", "predAbs": "mean", "T_query" : "mean"})
    
    test_dataset.to_csv("dataDerivative\\prediction\\pred" + str(queryID) + ".csv", header = True, index = True)
    return

Parallel(n_jobs=num_cores, verbose = 10)(delayed(predictBatch)(queryID) for queryID in IDs)

print("Total time elapsed: {0:.0f}:{1:02.0f} minutes\n".format(np.floor((time() - start)/60), (time()-start)%60))

#%%
i = 1
for ID in IDs:
    batchData["reference"] = ID
    print("Processing Reference Batch {0} - {1}/{2}".format(ID, i, n_batches))
    
    res = dtw(jsonObj = batchData, 
              open_ended = True, 
              all_subseq = True, 
              only_distance = True, 
              dist_measure = "euclidean",
              scale = True,
              shapeDTW = True,
              shape_descriptor = "derivative",
              n_jobs = 1)
    
    res.ApplyDTW()
    
    historicalDataset = pd.DataFrame(columns = columns)
    for queryID, output in res.output.items():
        
        historicalDataset = historicalDataset.append(pd.DataFrame(output, columns = columns), ignore_index = True)
    i += 1

    historicalDataset.to_csv("dataDerivative\\historical\\historicalDataset{}.csv".format(ID), header = True, index = False)

    print("Time elapsed: {0:.0f}:{1:02.0f} minutes\n".format(np.floor((time() - start)/60), (time()-start)%60))

print("Total time elapsed: {0:.0f}:{1:02.0f} minutes\n".format(np.floor((time() - start)/60), (time()-start)%60))

#%%

complete = pd.DataFrame(columns = columns)

for filename in os.listdir("dataDerivative\\historical"):
    if filename.startswith("historicalDataset"):
        complete = complete.append(pd.read_csv("dataDerivative\\historical\\"+filename, header = 0), ignore_index = True)
    
complete.drop_duplicates(inplace = True)    
complete.to_csv("dataDerivative\\historical\\completeDataset.csv", header = True, index = False)

#%%


train_columns = ["t_query", "t_ref", "T_ref", "DTW_dist"]
target = ["T_query"]

model = GradientBoostingRegressor()

for queryID in IDs:
    print("Predicting on batch {}".format(queryID))
    start = time()
    train_dataset = complete.loc[(complete.queryID != queryID) & (complete.refID != queryID), :]
    test_dataset = complete.loc[complete.queryID == queryID , :]
    
    X = train_dataset.loc[:, train_columns].values
    y = train_dataset.loc[:, target].values.reshape(-1,)
        
    model.fit(X,y)
# =============================================================================
#     filename = 'model.sav'
#     pickle.dump(model, open(filename, 'wb'))
# =============================================================================
    X_test = test_dataset.loc[:, train_columns].values
    
    test_dataset.loc[:, "predGB"] = model.predict(X_test)
    test_dataset.loc[:, "predAbs"] = test_dataset["T_ref"] - test_dataset["t_ref"] + test_dataset["t_query"] 
    test_dataset = test_dataset.astype({
                                      "refID" : int, 
                                      "queryID" : int,
                                      "t_query": int, 
                                      "t_ref": int, 
                                      "T_ref": int, 
                                      "DTW_dist": float, 
                                      "T_query": int,
                                      "predGB": float,
                                      "predAbs": int})

    test_dataset = test_dataset.groupby(["t_query"]).aggregate({"predGB" : "mean", "predAbs": "mean", "T_query" : "mean"})
    print("Time for prediction: {:.0f} seconds\n".format(time()-start))
    test_dataset.to_csv("dataDerivative\\prediction\\pred" + str(queryID) + ".csv", header = True, index = True)