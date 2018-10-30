# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 13:26:36 2018

@author: DEPAGRA
"""

import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from collections import defaultdict
from copy import deepcopy

class dtw:
    def __init__(self, jsonObj, open_ended = False, all_subseq = True, only_distance = True, dist_measure = "euclidean", n_jobs = 1):
        """
        Initialization of the class.
        open_ended: boolean
        all_subseq: boolean. If True, and if open_ended is True, runs the open ended version of the algorithm for all the subsequences of the query time series
        only_distance: boolean. If False, returns the warped time series too.
        dist_measure: string, the distance measure to use in the calculation of the cross-distance matrix
        n_jobs: integer. The number of cores to use in the computation of the cross-distance matrix. if -2, uses all but one cores.
        """
        
        self.all_subseq = open_ended and all_subseq # makes no sense to calculate for all subsequences if open_ended is not performed
        self.open_ended = open_ended
        self.n_jobs = n_jobs
        self.dist_measure = dist_measure
        
        self.output = defaultdict(list)
        
    
        self.ConvertDataFromJson(jsonObj)
        
        self.ApplyDTW()
    
    
    def ConvertDataFromJson(self, jsonObj):
        self.refID = jsonObj["reference"]
        self.reference = jsonObj[self.refID]
        self.queries = [{key:batch} for key, batch in jsonObj.items() if key != "reference" and key != self.refID]
        
    def ConvertToMVTS(self, batch):     # MVTS = Multi Variate Time Series
        d = len(batch)
        L = len(batch[0]['values'])
        
        MVTS = np.zeros((L, d))
        
        for (i, pv) in zip(np.arange(d), batch):
            MVTS[:, i] = pv['values']

        return MVTS        
        
    def ApplyDTW(self):
        
        self.referenceTS = self.ConvertToMVTS(self.reference)        
        
        prog = 0
        tot = len(self.queries)
        
        for queryBatch in self.queries:
            prog+=1
            print("\r{:>3.0%} Completed".format(prog/tot), end = "")
            for queryID, batch in queryBatch.items():
                self.queryID = queryID
                
                self.queryTS = self.ConvertToMVTS(batch)
                
                self.DTW()
    
                self.Results()
            
        print("\n")
                
    def DTW(self):
        
        self.N, d1 = self.referenceTS.shape    
        self.M, d2 = self.queryTS.shape
        
        if d1!= d2:
            print("Number of features not coherent between reference ({0}) and query ({1})".format(d1,d2))
            return
        
        self.d = d1  # d = dimensionality/number of features
        
        self.distanceMatrix = pairwise_distances(X = self.referenceTS, Y = self.queryTS, metric = self.dist_measure, n_jobs= self.n_jobs)
        
        self.AccumulatedDistanceComputation(step_pattern = "symmetric2")
    
        
    
    
    def AccumulatedDistanceComputation(self, step_pattern):
        
        self.accumulatedDistanceMatrix = np.zeros((self.N, self.M))
        self.accumulatedDistanceMatrix[0, 0] = self.distanceMatrix[0, 0]
        
        if step_pattern == "symmetric2":
            # first row
            for j in np.arange(1, self.M):
                self.accumulatedDistanceMatrix[0, j] = self.accumulatedDistanceMatrix[0, j - 1] + self.distanceMatrix[0, j]
             # first column
            for i in np.arange(1, self.N):
                self.accumulatedDistanceMatrix[i, 0] = self.accumulatedDistanceMatrix[i - 1, 0] + self.distanceMatrix[i, 0]
            
            # inside the matrix
            for i in np.arange(1, self.N):
                for j in np.arange(1, self.M):
                    self.accumulatedDistanceMatrix[i, j] = min(self.accumulatedDistanceMatrix[i-1, j], self.accumulatedDistanceMatrix[i, j-1], self.accumulatedDistanceMatrix[i-1, j-1] + self.distanceMatrix[i,j]) + self.distanceMatrix[i,j]
                    
                    
    def Results(self):
        if self.all_subseq:
            
            dataPoint = {"refID": self.refID,
                         "queryID": self.queryID,
                         "T_ref": self.N,
                         "T_query": self.M}
            
            for j in np.arange(self.M):
                
                dataPoint["t_query"] = j + 1
                dataPoint["t_ref"] = np.argmin(self.accumulatedDistanceMatrix[:, j]) + 1 
                norm_const = dataPoint["t_query"] + dataPoint["t_ref"]
                dataPoint["DTW_dist"] = self.accumulatedDistanceMatrix[dataPoint["t_ref"]-1, j] / norm_const
                self.output[self.queryID].append(deepcopy(dataPoint))

    
            