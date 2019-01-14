# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 13:26:36 2018

@author: DEPAGRA
"""

import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from collections import defaultdict
from copy import deepcopy
from scipy.signal import butter, lfilter


def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y


class dtw:
    def __init__(self, jsonObj, open_ended = False, all_subseq = True, only_distance = True, dist_measure = "euclidean", scale = True, low_filter = False, shapeDTW = False, n_points_shape = 3, shape_descriptor = "raw", norm_dist = True, undersampling = False, undersampling_factor = 1, n_jobs = 1):
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
        self.scale = scale
        self.low_filter = low_filter
        self.norm_dist = norm_dist
        self.shapeDTW = shapeDTW
        
        self.undersampling = undersampling
        self.undersampling_factor = self.undersampling*(undersampling_factor - 1) + 1
        
        self.n_points = n_points_shape
        
        if self.n_points%2 == 0:
            self.n_points += 1
        
        self.shape_descriptor = shape_descriptor
        
        self.output = defaultdict(list)
        
    
        self.ConvertDataFromJson(jsonObj)
    
    
    def ConvertDataFromJson(self, jsonObj):
        self.refID = jsonObj["reference"]
        self.reference = jsonObj[self.refID]
        self.queries = [{key:batch} for key, batch in jsonObj.items() if key != "reference" and key != self.refID]
        
    def ConvertToMVTS(self, batch):     # MVTS = Multi Variate Time Series
        """ 
        Takes a batch in the usual form (list of one dictionary per PV) and transforms it to a numpy array to perform calculations faster
        """
        if self.undersampling:
            self.L = np.ceil(len(batch[0]['values']) / self.undersampling_factor)
        else:
            self.L = len(batch[0]['values'])
        
        if not self.shapeDTW:
            self.d = len(batch)
            
            MVTS = np.zeros((self.L, self.d))
            
            for (i, pv) in zip(np.arange(self.d), batch):
                MVTS[:, i] = self.Scale(self.LowPassFilter(pv['values'][::self.undersampling_factor]))
                
        
        else:
            d_original = len(batch)
            if self.shape_descriptor == "raw":
                d_reshaped = d_original*self.n_points
                MVTS = np.zeros((self.L, d_reshaped))
            
                for (i, pv) in zip(np.arange(d_original), batch):
                    MVTS[:, i*self.n_points:(i+1)*self.n_points] = self.ShapeDescriptor(self.Scale(pv['values'][::self.undersampling_factor]))
            elif self.shape_descriptor == "derivative":
                d_reshaped = d_original
                MVTS = np.zeros((self.L, d_reshaped))
                for (i, pv) in zip(np.arange(d_original), batch):
                    MVTS[:, i] = self.ShapeDescriptor(self.Scale(pv['values'][::self.undersampling_factor]))
                
                
            
                
                
        return MVTS        
        
    def ApplyDTW(self):
        """
        Perform the algorithm for all the query time series present in the .queries attribute with respect to the single one reference time series. Should be not difficult to extend to more than one reference
        """
        
        self.referenceTS = self.ConvertToMVTS(self.reference)        
        
        progress = 0
        tot = len(self.queries)
        
        for queryBatch in self.queries:
            progress += 1
            print("\r{:>3.0%} Completed".format(progress/tot), end = "")
            for queryID, batch in queryBatch.items():
                self.queryID = queryID
                
                self.queryTS = self.ConvertToMVTS(batch)
                
                self.DTW()
    
                self.Results()
            
        print("\n")
                
    def DTW(self):
        """
        Standard DTW algorithm, the result is to create the distance matrix and the accumulated distance matrix. The .Results method returns the information required based on the input parameters
        """
        
        self.N, d1 = self.referenceTS.shape    
        self.M, d2 = self.queryTS.shape
        
        if d1!= d2:
            print("Number of features not coherent between reference ({0}) and query ({1})".format(d1,d2))
            return
        
        self.d = d1  # d = dimensionality/number of features
        
        self.distanceMatrix = pairwise_distances(X = self.referenceTS, Y = self.queryTS, metric = self.dist_measure, n_jobs= self.n_jobs)
        
        self.AccumulatedDistanceComputation(step_pattern = "symmetric2")
    
        
    def AccumulatedDistanceComputation(self, step_pattern):
        """
        Compute the accumulated distance matrix based on the given step_pattern
        """
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
        """
        Return the information in the form required via the initialization parameters. For now, just open-ended version on all subsequences
        """
        if self.all_subseq:
            
            dataPoint = {"refID": self.refID,
                         "queryID": self.queryID,
                         "T_ref": self.N,
                         "T_query": self.M}
            
            for j in np.arange(self.M):
                
                dataPoint["t_query"] = j + 1
                dataPoint["t_ref"] = np.argmin(self.accumulatedDistanceMatrix[:, j]) + 1 
                norm_const = dataPoint["t_query"] + dataPoint["t_ref"]
                dataPoint["DTW_dist"] = self.accumulatedDistanceMatrix[dataPoint["t_ref"]-1, j]/(1 + self.norm_dist*(norm_const - 1))
                self.output[self.queryID].append(deepcopy(dataPoint))
                
        elif (not self.open_ended) and (not self.all_subseq):
            norm_const = self.N + self.M
            
            dataPoint = {"refID": self.refID,
                         "queryID": self.queryID,
                         "T_ref": self.N,
                         "T_query": self.M,
                         "DTW_dist": self.accumulatedDistanceMatrix[self.N-1, self.M-1]/(1 + self.norm_dist*(norm_const - 1))}
            
            self.output[self.queryID].append(deepcopy(dataPoint))
                
    def ShapeDescriptor(self, ts):
        if self.shape_descriptor == "raw":
            reshapedTS = np.zeros((self.L, self.n_points))
            
            highIdx = (self.n_points - 1)//2
    
            reshapedTS[:, (self.n_points - 1)//2] = ts
            
            for shift, j in zip(np.arange(highIdx,0, -1), np.arange(0, highIdx)):
                reshapedTS[shift:, j] = ts[:-shift]
                reshapedTS[:j,j] = np.repeat(ts[0], j)
                
            for shift, j in zip(np.arange(1, highIdx + 1), np.arange(-highIdx, 0)):
                reshapedTS[:-shift, j] = ts[shift:]
                reshapedTS[-shift:,j] = np.repeat(ts[-1], shift)
                
            for i in np.arange(self.L):
                reshapedTS[i, :] = reshapedTS[i, :] - min(reshapedTS[i, :])
                
        
        
        elif self.shape_descriptor == "derivative":
            reshapedTS = np.zeros(ts.shape)
            
            for i in np.arange(1, len(reshapedTS) - 1):
                reshapedTS[i] = ((ts[i] - ts[i-1]) + ((ts[i+1] - ts[i-1])/2))/2
        
        return reshapedTS
            
            
    
    def Scale(self, pv):
        if self.scale:
            minVal, maxVal  = min(pv), max (pv)
            rangeVal = max(maxVal-minVal, 1e-7)
            return (np.array(pv)-minVal)/rangeVal
        return pv
    
    def LowPassFilter(self, ts, cutoff = 5, fs = 60.0, order = 6):
        if self.low_filter:
            return butter_lowpass_filter(ts, cutoff, fs, order)
                
        return ts
        