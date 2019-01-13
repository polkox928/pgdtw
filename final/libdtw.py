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
        """
        Returns a dictionary containing all the data, organized as:
        refID: the ID of the reference batch
        reference: reference batch in the usual format (list of dictionaries)
        queries: list of dictionaries in which the keys are the query batch's ID and the values are the actual batches (list of dictionaries)
        num_queries: number of query batches in the data set
        """
        refID = jsonObj["reference"]
        reference = jsonObj[refID]
        queries = {key:batch for key, batch in jsonObj.items() if key != "reference" and key != refID}

        return {"refID": refID,
                "reference": reference,
                "queries": queries,
                "num_queries": len(queries),
                "warpings" : dict(),
                "distances": dict()}

    def ConvertToMVTS(self, batch):     # MVTS = Multi Variate Time Series
        """
        Takes one batch in the usual form (list of one dictionary per PV) and transforms it to a numpy array to perform calculations faster
        """
        L = len(batch[0]['values']) # Length of a batch (number of data points per single PV)
        d = len(batch) # Number of PVs

        MVTS = np.zeros((L, d))

        for (i, pv) in zip(np.arange(d), batch):
            MVTS[:, i] = pv['values']

        return MVTS

    def CompDistMatrix(self, referenceTS, queryTS, dist_measure = "euclidean", n_jobs = 1):
        """
        Computes the distance matrix with N (length of the reference) number of rows and M (length of the query) number of columns (OK with convention on indices in DTW) with dist_measure as local distance measure

        referenceTS: MVTS representation of reference batch
        queryTS: MVTS representation of query batch
        dist_measure: string indicating the local distance measure to be used. Must be allowed by pairwise_distances
        n_jobs: number of jobs for pairwise_distances function. It could cause problems on windows
        """
        N, d1 = referenceTS.shape
        M, d2 = queryTS.shape

        if d1!= d2:
            print("Number of features not coherent between reference ({0}) and query ({1})".format(d1,d2))
            return

        d = d1  # d = dimensionality/number of features/PVs

        distanceMatrix = pairwise_distances(X = referenceTS, Y = queryTS, metric = dist_measure, n_jobs= n_jobs)

        return distanceMatrix
        #self.AccumulatedDistanceComputation(step_pattern = "symmetric2")

    def CompAccDistmatrix(self, distance_matrix, step_pattern):
        """
        Computes the accumulated distance matrix starting from the distance_matrix according to the step_pattern indicated
        distance_matrix: cross distance matrix
        step_pattern: string indicating the step pattern to be used. Can be symmetric1/2, symmetricP05 or symmetricPX, with X any positive integer
        """
        N, M = distance_matrix.shape
        accDistMatrix = np.zeros((N,M))

        for i in np.arange(N):
            for j in np.arange(M):
                accDistMatrix[i, j] = self.accElement(i, j, accDistMatrix, distance_matrix, step_pattern)

        return accDistMatrix

    def accElement(self, i, j, acc_dist_matrix, distance_matrix, step_pattern):
        """
        Computes the value of a cell of the accumulated distance matrix
        i: row (reference) index
        j: column (query) index
        acc_dist_matrix: current accumulated distance matrix
        distance_matrix: cross distance matrix
        step_pattern: step pattern to be used for calculations
        """
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

        patt = re.compile("symmetricP[1-9]+\d*")
        if patt.match(step_pattern):
            P = int(step_pattern[10:])
            p1 = acc_dist_matrix[i-P, j-(P+1)] + 2*sum([distance_matrix[i-p, j-(p+1)] for p in np.arange(0, P)]) + distance_matrix[i, j] if (i-P>=0 and j-(P+1)>=0) else np.inf
            p2 = acc_dist_matrix[i-1, j-1] + 2 * distance_matrix[i, j] if (i-1>=0 and j-1>=0) else np.inf
            p3 = acc_dist_matrix[i-(P+1), j-P] + 2*sum([distance_matrix[i-(p+1), j-p] for p in np.arange(0, P)]) + distance_matrix[i, j] if (i-(P+1)>=0 and j-P>=0) else np.inf

            return min(p1, p2, p3)

    def GetWarpingPath(self, acc_dist_matrix, step_pattern):
        N, M = acc_dist_matrix.shape
        warpingPath = list()

        if step_pattern == "symmetric1" or step_pattern == "symemtric2":
            i = N-1
            j = M-1
            while i != 0 or j != 0:
                warpingPath.append((i,j))
                candidates = list()
                if i > 0: candidates.append((acc_dist_matrix[i-1, j], (i-1,j)))
                if j > 0: candidates.append((acc_dist_matrix[i, j-1], (i, j-1)))
                if len(candidates) == 2: candidates.append((acc_dist_matrix[i-1,j-1], (i-1,j-1)))

                nextStep = min(candidates)[1]
                i,j = nextStep
            warpingPath.append((0,0))

            return warpingPath[::-1]

        elif step_pattern == "symmetricP05":
            #maxWarp = 2
            #minDiag = 1
            i = N-1
            j = M-1
            hStep = 0 #horizontal step
            vStep = 0 #vertical step
            dStep = 0 #diagonal step

            while i != 0 or j != 0:
                warpingPath.append((i,j))
                candidates = list()

                if hStep > 0:
                    if hStep == 1:
                        if j > 0: candidates.append((acc_dist_matrix[i, j-1], (i, j-1)))
                        if j > 0 and i > 0: candidates.append((acc_dist_matrix[i-1, j-1], (i-1, j-1)))
                    elif hStep == 2:
                        if j > 0 and i > 0: candidates.append((acc_dist_matrix[i-1, j-1], (i-1, j-1)))

                elif vStep > 0:
                    if vStep == 1:
                        if i > 0: candidates.append((acc_dist_matrix[i-1, j], (i-1, j)))
                        if j > 0 and i > 0: candidates.append((acc_dist_matrix[i-1, j-1], (i-1, j-1)))
                    elif vStep == 2:
                        if j > 0 and i > 0: candidates.append((acc_dist_matrix[i-1, j-1], (i-1, j-1)))

                else:
                    if j > 0: candidates.append((acc_dist_matrix[i, j-1], (i, j-1)))
                    if i > 0: candidates.append((acc_dist_matrix[i-1, j], (i-1, j)))
                    if j > 0 and i > 0: candidates.append((acc_dist_matrix[i-1, j-1], (i-1, j-1)))

                nextStep = min(candidates)[1]
                v = nextStep[0] < i
                h = nextStep[1] < j
                d = v and h

                if d:
                    vStep = 0
                    hStep = 0
                elif v: vStep += 1
                elif h: hStep += 1

                i,j = nextStep

            warpingPath.append((0,0))

            return warpingPath[::-1]


        else:
            patt = re.compile("symmetricP[1-9]+\d*")
            if patt.match(step_pattern):

                minDiagSteps = int(step_pattern[10:])
                wStep = 0
                dStep = 0
                i = N-1
                j = M-1

                while i != 0  and j != 0:
                    warpingPath.append((i,j))
                    candidates = list()
                    if wStep > 0: candidates.append((acc_dist_matrix[i-1,j-1], (i-1,j-1)))
                    else:
                        if j > 0: candidates.append((acc_dist_matrix[i, j-1], (i, j-1)))
                        if i > 0: candidates.append((acc_dist_matrix[i-1, j], (i-1, j)))
                        if len(candidates) == 2: candidates.append((acc_dist_matrix[i-1, j-1], (i-1, j-1)))

                    nextStep = min(candidates)[1]
                    v = nextStep[0] < i
                    h = nextStep[1] < j
                    d = v and h

                    if d:
                        dStep += 1
                        if dStep == minDiagSteps:
                            dStep = 0
                            wStep = 0
                        elif dStep < minDiagSteps and wStep > 0:
                            pass
                        elif dStep < minDiagSteps and wStep == 0:
                            dStep = 0
                    else:
                        wStep += 1

                    i, j = nextStep

                warpingPath.append((0,0))

                return warpingPath[::-1]

            else: print("Invalid step-pattern")

    def CallDTW(self, queryID, step_pattern = "symmetricP05", dist_measure = "euclidean", n_jobs = 1, results = False):
        referenceTS = self.ConvertToMVTS(self.data.reference)
        queryTS = self.ConvertToMVTS(self.data.queries[queryID])

        result = DTW(referenceTS, queryTS, step_pattern, dist_measure, n_jobs)

        self.data["warpings"][queryID] = result["warping"]
        self.data["distances"][queryID] = result["DTW_distance"]

        if results:
            return result

    def DTW(self, referenceTS, queryTS, step_pattern = "symmetricP05", dist_measure = "euclidean", n_jobs = 1):

        distanceMatrix = self.CompDistMatrix(referenceTS, queryTS, dist_measure, n_jobs)

        accDistMatrix = self.CompAccDistmatrix(distanceMatrix, step_pattern)

        warping = self.GetWarpingPath(accDistMatrix, step_pattern)

        dtwDist = accDistMatrix[-1,-1]



        return {"warping": warping,
                "DTW_distance": dtwDist}
