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
    def __init__(self, jsonObj, open_ended = False, all_subseq = True, only_distance = True, dist_measure = "euclidean", scale = True, n_jobs = -2):
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
        self.d = len(batch)
        self.L = len(batch[0]['values'])

        MVTS = np.zeros((self.L, self.d))

        for (i, pv) in zip(np.arange(self.d), batch):
            MVTS[:, i] = pv['values']
            if self.scale:
                minVal, maxVal  = min(MVTS[:, i]), max (MVTS[:, i])
                rangeVal = max(maxVal-minVal, 1e-7)
                MVTS[:, i] = (MVTS[:, i]-minVal)/rangeVal

        return MVTS

    def ApplyDTW(self):
        """
        Perform the algorithm for all the query time series present in the .queries attribute with respect to the single one reference time series. Should be not difficult to extend to more than one reference
        """

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
                dataPoint["DTW_dist"] = self.accumulatedDistanceMatrix[dataPoint["t_ref"]-1, j] / norm_const
                self.output[self.queryID].append(deepcopy(dataPoint))

    def shapeDescriptor(self, ts, descriptor = "raw", n_points = 3):
        nd = self.d * n_points # dimensionality of the reshaped ts
        shapedTS = np.zeros((self.L, nd))
# =============================================================================
#         for i in np.arange(self.L):
#             for j in np.arange(self.d):
#                 for z in np.arange(-1, n_points - 1):
#                     try:
#                         pass
#                     except:
#                         pass
# =============================================================================