import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
import re

class dtw:
    def __init__(self):#, jsonObj = 0):
        """
        Initialization of the class.
        jsonObj: contains the data in the usual format
        """
        #self.data = self.ConvertDataFromJson(jsonObj)
        pass

    def ConvertDataFromJson(self, jsonObj):
        refID = jsonObj["reference"]
        reference = jsonObj[refID]
        queries = [{key:batch} for key, batch in jsonObj.items() if key != "reference" and key != refID]

        return {"refID": refID,
                "reference": reference,
                "queries": queries,
                "num_queries": len(queries)}

    def ConvertToMVTS(self, batch):     # MVTS = Multi Variate Time Series
        """
        Takes a batch in the usual form (list of one dictionary per PV) and transforms it to a numpy array to perform calculations faster
        """
        distance_matrix = len(batch[0]['values'])
        L = len(batch)

        MVTS = np.zeros((L, distance_matrix))

        for (i, pv) in zip(np.aranacc_dist_matrixe(distance_matrix), batch):
            MVTS[:, i] = pv['values']

        return MVTS

    def CompDistMatrix(self, referenceTS, queryTS, dist_measure, n_jobs):
        N, d1 = referenceTS.shape
        M, d2 = queryTS.shape

        if d1!= d2:
            print("Number of features not coherent between reference ({0}) and query ({1})".format(d1,d2))
            return

        distance_matrix = d1  # distance_matrix = dimensionality/number of features

        distanceMatrix = pairwise_distances(X = referenceTS, Y = queryTS, metric = dist_measure, n_jobs= n_jobs)

        return distanceMatrix
        #self.AccumulatedDistanceComputation(step_pattern = "symmetric2")

    def CompAccDistmatrix(self, distance_matrix, step_pattern):
        accDistMatrix = np.zeros(distance_matrix.shape)
        N, M = distance_matrix.shape

        for i in np.arange(N):
            for j in np.arange(M):
                accDistMatrix[i, j] = self.accElement(i, j, accDistMatrix, distance_matrix, step_pattern)

        return accDistMatrix

    def accElement(self, i, j, acc_dist_matrix, distance_matrix, step_pattern):
        if (i==0 and j==0): return distance_matrix[0, 0]

        if step_pattern == "symmetricP05":

            p1 = acc_dist_matrix[i-1, j-3] + 2 * distance_matrix[i, j-2] + distance_matrix[i, j-1] +     distance_matrix[i, j] if (i-1>=0 and j-3 >=0) else np.inf
            p2 = acc_dist_matrix[i-1, j-2] + 2 * distance_matrix[i, j-1] + distance_matrix[i, j] if (i-1>=0 and j-2>=0) else np.inf
            p3 = acc_dist_matrix[i-1, j-1] + 2 * distance_matrix[i, j] if (i-1>=0 and j-1>=0) else np.inf
            p4 = acc_dist_matrix[i-2, j-1] + 2 * distance_matrix[i-1, j] + distance_matrix[i, j] if (i-2>=0 and j-1>=0) else np.inf
            p5 = acc_dist_matrix[i-3, j-1] + 2 * distance_matrix[i-2, j] + distance_matrix[i-1, j] +     distance_matrix[i, j] if (i-3>=0 and j-1 >=0) else np.inf

            return min(p1, p2, p3, p4, p5)

        if step_pattern == "symmetric1":
            p1 = acc_dist_matrix[i, j-1] + distance_matrix[i, j] if (j-1>=0) else np.inf
            p2 = acc_dist_matrix[i-1, j-1] + distance_matrix[i, j] if (i-1>=0 and j-1>=0) else np.inf
            p3 = acc_dist_matrix[i-1, j] + distance_matrix[i, j] if (i-1>=0) else np.inf

            return min(p1, p2, p3)

        if step_pattern == "symmetric2":
            p1 = acc_dist_matrix[i, j-1] + distance_matrix[i, j] if (j-1>=0) else np.inf
            p2 = acc_dist_matrix[i-1, j-1] + 2 * distance_matrix[i, j] if (i-1>=0 and j-1>=0) else np.inf
            p3 = acc_dist_matrix[i-1, j] + distance_matrix[i, j] if (i-1>=0) else np.inf

            return min(p1, p2, p3)

        patt = re.compile("symmetricP[1-9]+\d+")
        if patt.match(step_pattern):
            P = int(step_pattern[10:])
            p1 = acc_dist_matrix[i-P, j-(P+1)] + sum([distance_matrix[i-p, j-(p+1)] for p in np.arange(0, P)]) if (i-P>=0 and j-(P+1)>=0) else np.inf
            p2 = acc_dist_matrix[i-1, j-1] + 2 * distance_matrix[i, j] if (i-1>=0 and j-1>=0) else np.inf
            p3 = acc_dist_matrix[i-(P+1), j-P] + sum([distance_matrix[i-(p+1), j-p] for p in np.arange(0, P)]) if (i-(P+1)>=0 and j-P>=0) else np.inf

            return min(p1, p2, p3)
