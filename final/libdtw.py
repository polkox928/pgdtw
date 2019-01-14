import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
import re
import matplotlib.pyplot as plt
import matplotlib

class dtw:
    def __init__(self, jsonObj = False):
        """
        Initialization of the class.
        jsonObj: contains the data in the usual format
        """
        if not jsonObj:    
            pass
        else:
            self.data = self.ConvertDataFromJson(jsonObj)
            self.scaleParams = self.GetScalingParameters()
            self.RmvConstFeat()

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
        
    def GetScalingParameters(self):
        """
        Computes the parameters necessary for scaling the features as a 'group'. This means considering the mean range of a variable across al the data set.
        This seems creating problems, since the distributions for the minimum and the maximum are too spread out. This method is here just in case of future use and to help removing non-informative (constant) features.
        avgRange = [avgMin, avgMax]  
        """
        scaleParams = dict()
            
        for pv in self.data['reference']:
            pvName = pv['name']
            pvMin = min(pv['values'])
            pvMax = max(pv['values'])
            
            scaleParams[pvName] = [[pvMin], [pvMax]]
        
        for _id, batch in self.data['queries'].items():
            for pv in batch:
                pvName = pv['name']
                pvMin = min(pv['values'])
                pvMax = max(pv['values'])
                
                scaleParams[pvName][0].append(pvMin)
                scaleParams[pvName][1].append(pvMax)
        
        pvNames = scaleParams.keys()
        for pv in pvNames:
            scaleParams[pv] = np.median(scaleParams[pv], axis = 1)
            
        return scaleParams
            
    def RmvConstFeat(self):
        """
        Removes non-informative features (features with low variability)
        """
        constFeats = list()
        for pvName, avgRange in self.scaleParams.items():
            if abs(avgRange[0]-avgRange[1]) < 1e-6:
                constFeats.append(pvName)
        
        IDs = list(self.data['queries'].keys())
        for _id in IDs:
            self.data['queries'][_id] = [pv for pv in self.data['queries'][_id] if pv['name'] not in constFeats]
            
        self.data['reference'] = [pv for pv in self.data['reference'] if pv['name'] not in constFeats]
        
    def ScalePV(self, pv_name, pv_values, mode = "single"):
        """
        Scales features in two possible ways:
            'single': the feature is scaled according to the values it assumes in the current batch
            'group': the feature is scaled according to its average range across the whole data set
        """
        if mode == "single":
            minPV = min(pv_values)
            maxPV = max(pv_values)
            if abs(maxPV-minPV) > 1e-6:
                scaledPvValues = (np.array(pv_values)-minPV)/(maxPV-minPV)
            else:
                scaledPvValues = .5 * np.ones(len(pv_values))
        elif mode == "group":
            avgMin, avgMax = self.scaleParams[pv_name]
            scaledPvValues = (np.array(pv_values)-avgMin)/(avgMax-avgMin)
        return scaledPvValues        
        
    def ConvertToMVTS(self, batch):     # MVTS = Multi Variate Time Series
        """
        Takes one batch in the usual form (list of one dictionary per PV) and transforms it to a numpy array to perform calculations faster
        """
        L = len(batch[0]['values']) # Length of a batch (number of data points per single PV)
        d = len(batch) # Number of PVs

        MVTS = np.zeros((L, d))

        for (i, pv) in zip(np.arange(d), batch):
            MVTS[:, i] = self.ScalePV(pv['name'], pv['values'], "single")

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

        #d = d1  # d = dimensionality/number of features/PVs

        distanceMatrix = pairwise_distances(X = referenceTS, Y = queryTS, metric = dist_measure, n_jobs= n_jobs)

        return distanceMatrix
        #self.AccumulatedDistanceComputation(step_pattern = "symmetric2")

    def CompAccDistMatrix(self, distance_matrix, step_pattern = 'symmetricP05'):
        """
        Computes the accumulated distance matrix starting from the distance_matrix according to the step_pattern indicated
        distance_matrix: cross distance matrix
        step_pattern: string indicating the step pattern to be used. Can be symmetric1/2, symmetricP05 or symmetricPX, with X any positive integer
        """
        N, M = distance_matrix.shape
        accDistMatrix = np.zeros((N,M))

        for i in np.arange(N):
            for j in np.arange(M):
                accDistMatrix[i, j] = self.CompAccElement(i, j, accDistMatrix, distance_matrix, step_pattern)

        return accDistMatrix

    def CompAccElement(self, i, j, acc_dist_matrix, distance_matrix, step_pattern):
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

    def GetWarpingPath(self, acc_dist_matrix, step_pattern, N, M):
        """
        Computes the warping path on the acc_dist_matrix induced by step_pattern starting from the (N,M) point (this in order to use the method in both open_ended and global alignment)
        Return the warping path (list of tuples) in ascending order
        """
        #N, M = acc_dist_matrix.shape
        warpingPath = list()

        if step_pattern == "symmetric1" or step_pattern == "symmetric2":
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
            if np.isinf(acc_dist_matrix[i,j]):
                print("Invalid value for P, a global alignment is not possible with this local constraint")
                return
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
               
                if np.isinf(acc_dist_matrix[i,j]):
                    print("Invalid value for P, a global alignment is not possible with this local constraint")
                    return
                
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

    def CallDTW(self, queryID, step_pattern = "symmetricP05", dist_measure = "euclidean", n_jobs = 1, open_ended = False, get_results = False):
        """
        Calls the DTW method on the data stored in the .data attribute (needs only the queryID in addition to standard parameters)
        get_results if True returns the distance and the warping calculated; if False, only the .data attribute is updated
        """
        referenceTS = self.ConvertToMVTS(self.data['reference'])
        queryTS = self.ConvertToMVTS(self.data['queries'][queryID])

        result = self.DTW(referenceTS, queryTS, step_pattern, dist_measure, n_jobs, open_ended)

        self.data["warpings"][queryID] = result["warping"]
        self.data["distances"][queryID] = result["DTW_distance"]

        if get_results:
            return result

    def DTW(self, referenceTS, queryTS, step_pattern = "symmetricP05", dist_measure = "euclidean", n_jobs = 1, open_ended = False):
        """
        Compute alignment betwwen referenceTS and queryTS (already in MVTS form).
        Separate from CallDTW() for testing purposes
        """
        # Check for coherence of local constraint and global alignment (in case a PX local constraint is used)
        if not open_ended:
            patt = re.compile("symmetricP[1-9]+\d*")
            if patt.match(step_pattern):
                P = int(step_pattern[step_pattern.index("P")+1:])
                N, M = len(referenceTS), len(queryTS)
                Pmax = np.floor(min(N,M)/np.abs(N-M))
                if P > Pmax:
                    print("Invalid value for P, a global alignment is not possible with this local constraint")
                    return
            else: pass

        distanceMatrix = self.CompDistMatrix(referenceTS, queryTS, dist_measure, n_jobs)

        accDistMatrix = self.CompAccDistMatrix(distanceMatrix, step_pattern)

        N, M = accDistMatrix.shape
        # In case of open-ended version, correctly identifies the starting point on the reference batch for warping
        if open_ended: 
            N = self.GetRefPrefixLength(accDistMatrix)

        warping = self.GetWarpingPath(accDistMatrix, step_pattern, N, M)

        dtwDist = accDistMatrix[N-1, M-1]

        return {"warping": warping,
                "DTW_distance": dtwDist}

    def GetRefPrefixLength(self, acc_dist_matrix):
        """
        Computes the length of the reference prefix in case of open-ended alignment
        """
        # In case of open-ended version, correctly identifies the starting point on the reference batch for warping
        refPrefixLen = np.argmin(acc_dist_matrix[:, -1]) + 1 
        return refPrefixLen
    
    def DistanceCostPlot(self, distance_matrix):
        cmap = matplotlib.cm.inferno
        cmap.set_bad('green',.5)
        masked_array = np.ma.array (distance_matrix, mask=np.isnan(distance_matrix))
        im = plt.imshow(masked_array, interpolation='nearest', cmap=cmap) 
        
        #ax.imshow(masked_array, interpolation='nearest', cmap=cmap)

        plt.gca().invert_yaxis()
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.grid()
        plt.colorbar();