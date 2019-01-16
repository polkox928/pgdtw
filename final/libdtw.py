"""
dtw class and loadData function
"""
from collections import defaultdict
import re
import pickle
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
import matplotlib.pyplot as plt
import matplotlib


class Dtw:
    """
    Everything related to DTW and experimentation
    """
    def __init__(self, json_obj=False):
        """
        Initialization of the class.
        json_obj: contains the data in the usual format
        """
        if not json_obj:
            pass
        else:
            self.data = self.convert_data_from_json(json_obj)
            self.scale_params = self.get_scaling_parameters()
            self.remove_const_feats()

    def convert_data_from_json(self, json_obj):
        """
        Returns a dictionary containing all the data, organized as:
        ref_id: the ID of the reference batch
        reference: reference batch in the usual format (list of dictionaries)
        queries: list of dictionaries in which the keys are the query batch's ID and the values are
        the actual batches (list of dictionaries)
        num_queries: number of query batches in the data set
        """
        ref_id = json_obj["reference"]
        reference = json_obj[ref_id]
        queries = {key: batch for key, batch in json_obj.items() if key !=
                   "reference" and key != ref_id}

        return {"ref_id": ref_id,
                "reference": reference,
                "queries": queries,
                "num_queries": len(queries),
                "warpings": dict(),
                "distances": dict(),
                "queriesID": list(queries.keys()),
                "time_distortion": defaultdict(dict),
                "distance_distortion": defaultdict(dict)}

    def get_scaling_parameters(self):
        """
        Computes the parameters necessary for scaling the features as a 'group'.
        This means considering the mean range of a variable across al the data set.
        This seems creating problems, since the distributions for the minimum and the
        maximum are too spread out. This method is here just in case of future use and to help
        removing non-informative (constant) features.
        avg_range = [avg_min, avg_max]
        """
        scale_params = dict()

        for pv_dict in self.data['reference']:
            pv_name = pv_dict['name']
            pv_min = min(pv_dict['values'])
            pv_max = max(pv_dict['values'])

            scale_params[pv_name] = [[pv_min], [pv_max]]

        for _id, batch in self.data['queries'].items():
            for pv_dict in batch:
                pv_name = pv_dict['name']
                pv_min = min(pv_dict['values'])
                pv_max = max(pv_dict['values'])

                scale_params[pv_name][0].append(pv_min)
                scale_params[pv_name][1].append(pv_max)

        pv_names = scale_params.keys()
        for pv_name in pv_names:
            scale_params[pv_name] = np.median(scale_params[pv_name], axis=1)

        return scale_params

    def remove_const_feats(self):
        """
        Removes non-informative features (features with low variability)
        """
        const_feats = list()
        for pv_name, avg_range in self.scale_params.items():
            if abs(avg_range[0]-avg_range[1]) < 1e-6:
                const_feats.append(pv_name)

        ids = list(self.data['queries'].keys())
        for _id in ids:
            self.data['queries'][_id] = [pv_dict for pv_dict in self.data['queries']
                                         [_id] if pv_dict['name'] not in const_feats]

        self.data['reference'] = [pv_dict for pv_dict in self.data['reference']
                                  if pv_dict['name'] not in const_feats]

    def scale_pv(self, pv_name, pv_values, mode="single"):
        """
        Scales features in two possible ways:
            'single': the feature is scaled according to the values it assumes in the current batch
            'group': the feature is scaled according to its average range across the whole data set
        """
        if mode == "single":
            pv_min = min(pv_values)
            pv_max = max(pv_values)
            if abs(pv_max-pv_min) > 1e-6:
                scaled_pv_values = (np.array(pv_values)-pv_min)/(pv_max-pv_min)
            else:
                scaled_pv_values = .5 * np.ones(len(pv_values))
        elif mode == "group":
            avg_min, avg_max = self.scale_params[pv_name]
            scaled_pv_values = (np.array(pv_values)-avg_min)/(avg_max-avg_min)
        return scaled_pv_values

    def convert_to_mvts(self, batch):     # MVTS = Multi Variate Time Series
        """
        Takes one batch in the usual form (list of one dictionary per PV) and transforms
        it to a numpy array to perform calculations faster
        """
        k = len(batch[0]['values'])  # Length of a batch (number of data points per single PV)
        d = len(batch)  # Number of PVs

        MVTS = np.zeros((k, d))

        for (i, pv_dict) in zip(np.arange(d), batch):
            MVTS[:, i] = self.scale_pv(pv_dict['name'], pv_dict['values'], "single")

        return MVTS

    def CompDistMatrix(self, referenceTS, queryTS, dist_measure="euclidean", n_jobs=1):
        """
        Computes the distance matrix with N (length of the reference) number of rows and M (length
        of the query) number of columns (OK with convention on indices in DTW) with dist_measure as
        local distance measure

        referenceTS: MVTS representation of reference batch
        queryTS: MVTS representation of query batch
        dist_measure: string indicating the local distance measure to be used.
                        Must be allowed by pairwise_distances
        n_jobs: number of jobs for pairwise_distances function. It could cause problems on windows
        """
        _, d1 = referenceTS.shape
        _, d2 = queryTS.shape

        if d1 != d2:
            print("Number of features not coherent between reference ({0}) and query ({1})"\
                                                                                  .format(d1, d2))
            return

        distanceMatrix = pairwise_distances(
            X=referenceTS, Y=queryTS, metric=dist_measure, n_jobs=n_jobs)

        return distanceMatrix

    def CompAccDistMatrix(self, distance_matrix, step_pattern='symmetricP05'):
        """
        Computes the accumulated distance matrix starting from the distance_matrix according to the
        step_pattern indicated
        distance_matrix: cross distance matrix
        step_pattern: string indicating the step pattern to be used. Can be symmetric1/2,
        symmetricP05 or symmetricPX, with X any positive integer
        """
        N, M = distance_matrix.shape
        accDistMatrix = np.empty((N, M))

        for i in np.arange(N):
            for j in np.arange(M):
                accDistMatrix[i, j] = self.CompAccElement(
                    i, j, accDistMatrix, distance_matrix, step_pattern)\
                                            if self.Itakura(i, j, N, M, step_pattern) else np.inf

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
        if (i == 0 and j == 0):
            return distance_matrix[0, 0]

        if step_pattern == "symmetricP05":

            p1 = acc_dist_matrix[i-1, j-3] + 2 * distance_matrix[i, j-2] + distance_matrix[i, j-1] \
                + distance_matrix[i, j] if (i-1 >= 0 and j-3 >= 0) else np.inf
            p2 = acc_dist_matrix[i-1, j-2] + 2 * distance_matrix[i, j-1] + \
                distance_matrix[i, j] if (i-1 >= 0 and j-2 >= 0) else np.inf
            p3 = acc_dist_matrix[i-1, j-1] + 2 * \
                distance_matrix[i, j] if (i-1 >= 0 and j-1 >= 0) else np.inf
            p4 = acc_dist_matrix[i-2, j-1] + 2 * distance_matrix[i-1, j] + \
                distance_matrix[i, j] if (i-2 >= 0 and j-1 >= 0) else np.inf
            p5 = acc_dist_matrix[i-3, j-1] + 2 * distance_matrix[i-2, j] + distance_matrix[i-1, j] \
                + distance_matrix[i, j] if (i-3 >= 0 and j-1 >= 0) else np.inf

            return min(p1, p2, p3, p4, p5)

        if step_pattern == "symmetric1":
            p1 = acc_dist_matrix[i, j-1] + distance_matrix[i, j] if (j-1 >= 0) else np.inf
            p2 = acc_dist_matrix[i-1, j-1] + distance_matrix[i, j]\
                                                            if (i-1 >= 0 and j-1 >= 0) else np.inf
            p3 = acc_dist_matrix[i-1, j] + distance_matrix[i, j] if (i-1 >= 0) else np.inf

            return min(p1, p2, p3)

        if step_pattern == "symmetric2":
            p1 = acc_dist_matrix[i, j-1] + distance_matrix[i, j] if (j-1 >= 0) else np.inf
            p2 = acc_dist_matrix[i-1, j-1] + 2 * \
                distance_matrix[i, j] if (i-1 >= 0 and j-1 >= 0) else np.inf
            p3 = acc_dist_matrix[i-1, j] + distance_matrix[i, j] if (i-1 >= 0) else np.inf

            return min(p1, p2, p3)

        patt = re.compile("symmetricP[1-9]+\d*")
        if patt.match(step_pattern):
            P = int(step_pattern[10:])
            p1 = acc_dist_matrix[i-P, j-(P+1)] + 2*sum([distance_matrix[i-p, j-(p+1)] for p in\
                np.arange(0, P)]) + distance_matrix[i, j] if (i-P >= 0 and j-(P+1) >= 0) else np.inf
            p2 = acc_dist_matrix[i-1, j-1] + \
                                    2 * distance_matrix[i, j] if (i-1 >= 0 and j-1 >= 0) else np.inf
            p3 = acc_dist_matrix[i-(P+1), j-P] + 2*sum([distance_matrix[i-(p+1), j-p] \
                                        for p in np.arange(0, P)]) + distance_matrix[i, j] \
                                                                if (i-(P+1) >= 0 and j-P >= 0) \
                                                                                        else np.inf

            return min(p1, p2, p3)

    def GetWarpingPath(self, acc_dist_matrix, step_pattern, N, M):
        """
        Computes the warping path on the acc_dist_matrix induced by step_pattern starting from
        the (N,M) point (this in order to use the method in both open_ended and global alignment)
        Return the warping path (list of tuples) in ascending order
        """
        #N, M = acc_dist_matrix.shape
        warpingPath = list()

        if step_pattern == "symmetric1" or step_pattern == "symmetric2":
            i = N-1
            j = M-1
            while i != 0 or j != 0:
                warpingPath.append((i, j))
                candidates = list()
                if i > 0:
                    candidates.append((acc_dist_matrix[i-1, j], (i-1, j)))
                if j > 0:
                    candidates.append((acc_dist_matrix[i, j-1], (i, j-1)))
                if len(candidates) == 2:
                    candidates.append((acc_dist_matrix[i-1, j-1], (i-1, j-1)))

                nextStep = min(candidates)[1]
                i, j = nextStep
            warpingPath.append((0, 0))

            return warpingPath[::-1]

        elif step_pattern == "symmetricP05":
            #maxWarp = 2
            #minDiag = 1
            i = N-1
            j = M-1
            if np.isinf(acc_dist_matrix[i, j]):
                print("Invalid value for P, \
                      a global alignment is not possible with this local constraint")
                return
            hStep = 0  # horizontal step
            vStep = 0  # vertical step
            dStep = 0  # diagonal step

            while i != 0 or j != 0:
                warpingPath.append((i, j))
                candidates = list()

                if hStep > 0:
                    if hStep == 1:
                        if j > 0:
                            candidates.append((acc_dist_matrix[i, j-1], (i, j-1)))
                        if j > 0 and i > 0:
                            candidates.append((acc_dist_matrix[i-1, j-1], (i-1, j-1)))
                    elif hStep == 2:
                        if j > 0 and i > 0:
                            candidates.append((acc_dist_matrix[i-1, j-1], (i-1, j-1)))

                elif vStep > 0:
                    if vStep == 1:
                        if i > 0:
                            candidates.append((acc_dist_matrix[i-1, j], (i-1, j)))
                        if j > 0 and i > 0:
                            candidates.append((acc_dist_matrix[i-1, j-1], (i-1, j-1)))
                    elif vStep == 2:
                        if j > 0 and i > 0:
                            candidates.append((acc_dist_matrix[i-1, j-1], (i-1, j-1)))

                else:
                    if j > 0:
                        candidates.append((acc_dist_matrix[i, j-1], (i, j-1)))
                    if i > 0:
                        candidates.append((acc_dist_matrix[i-1, j], (i-1, j)))
                    if j > 0 and i > 0:
                        candidates.append((acc_dist_matrix[i-1, j-1], (i-1, j-1)))

                nextStep = min(candidates)[1]
                v = nextStep[0] < i
                h = nextStep[1] < j
                d = v and h

                if d:
                    vStep = 0
                    hStep = 0
                elif v:
                    vStep += 1
                elif h:
                    hStep += 1

                i, j = nextStep

            warpingPath.append((0, 0))

            return warpingPath[::-1]

        else:
            patt = re.compile("symmetricP[1-9]+\d*")
            if patt.match(step_pattern):

                minDiagSteps = int(step_pattern[10:])

                wStep = 0
                dStep = 0
                i = N-1
                j = M-1

                if np.isinf(acc_dist_matrix[i, j]):
                    print("Invalid value for P, \
                          a global alignment is not possible with this local constraint")
                    return

                while i != 0 and j != 0:
                    warpingPath.append((i, j))
                    candidates = list()
                    if wStep > 0:
                        candidates.append((acc_dist_matrix[i-1, j-1], (i-1, j-1)))
                    else:
                        if j > 0:
                            candidates.append((acc_dist_matrix[i, j-1], (i, j-1)))
                        if i > 0:
                            candidates.append((acc_dist_matrix[i-1, j], (i-1, j)))
                        if len(candidates) == 2:
                            candidates.append((acc_dist_matrix[i-1, j-1], (i-1, j-1)))

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

                warpingPath.append((0, 0))

                return warpingPath[::-1]

            else:
                print("Invalid step-pattern")

    def CallDTW(self, queryID, step_pattern="symmetricP05", dist_measure="euclidean",\
                                                    n_jobs=1, open_ended=False, get_results=False):
        """
        Calls the DTW method on the data stored in the .data attribute (needs only the queryID in \
        addition to standard parameters)
        get_results if True returns the distance and the warping calculated; if False, \
        only the .data attribute is updated
        """
        referenceTS = self.convert_to_mvts(self.data['reference'])
        queryTS = self.convert_to_mvts(self.data['queries'][queryID])

        result = self.DTW(referenceTS, queryTS, step_pattern, dist_measure, n_jobs, open_ended)

        self.data["warpings"][queryID] = result["warping"]
        self.data["distances"][queryID] = result["DTW_distance"]
        self.data['time_distortion'][step_pattern][queryID] = self.TimeDistortion(result['warping'])
        self.data['distance_distortion'][step_pattern][queryID] = result["DTW_distance"]

        if get_results:
            return result

    def DTW(self, referenceTS, queryTS, step_pattern="symmetricP05", dist_measure="euclidean",\
                                                                        n_jobs=1, open_ended=False):
        """
        Compute alignment betwwen referenceTS and queryTS (already in MVTS form).
        Separate from CallDTW() for testing purposes
        """
        # Check for coherence of local constraint and global alignment
        # (in case a PX local constraint is used)
        if not open_ended:
            patt = re.compile("symmetricP[1-9]+\d*")
            if patt.match(step_pattern):
                P = int(step_pattern[step_pattern.index("P")+1:])
                N, M = len(referenceTS), len(queryTS)
                Pmax = np.floor(min(N, M)/np.abs(N-M)) if np.abs(N-M) > 0 else np.inf
                if P > Pmax:
                    print("Invalid value for P, \
                                  a global alignment is not possible with this local constraint")
                    return
            else:
                pass

        distanceMatrix = self.CompDistMatrix(referenceTS, queryTS, dist_measure, n_jobs)

        accDistMatrix = self.CompAccDistMatrix(distanceMatrix, step_pattern)

        N, M = accDistMatrix.shape
        # In case of open-ended version
        # correctly identifies the starting point on the reference batch for warping
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
        # In case of open-ended version
        # correctly identifies the starting point on the reference batch for warping
        refPrefixLen = np.argmin(acc_dist_matrix[:, -1]) + 1
        return refPrefixLen

    def DistanceCostPlot(self, distance_matrix):
        """
        Draws a heatmap of distance_matrix, nan values are colored in green
        """
        cmap = matplotlib.cm.inferno
        cmap.set_bad('green', .3)
        masked_array = np.ma.array(distance_matrix, mask=np.isnan(distance_matrix))
        im = plt.imshow(masked_array, interpolation='nearest', cmap=cmap)

        #ax.imshow(masked_array, interpolation='nearest', cmap=cmap)

        plt.gca().invert_yaxis()
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.grid()
        plt.colorbar()

    def TimeDistortion(self, warping_path):
        T = len(warping_path)
        fq = [w[1] for w in warping_path]
        fr = [w[0] for w in warping_path]

        td = [(fr[t+1] - fr[t])*(fq[t+1] - fq[t]) == 0 for t in np.arange(T-1)]

        return sum(td)

    def AvgTimeDistortion(self, step_pattern):
        if len(self.data['time_distortion'][step_pattern]) != self.data['num_queries']:
            print('Not every query aligned, align the remaining queries')
            return
        else:
            I = self.data['num_queries']
            avgTD = 1/I*sum(self.data['time_distortion'][step_pattern].values())

            return avgTD

    def AvgDistance(self, step_pattern):
        if len(self.data['distance_distortion'][step_pattern]) != self.data['num_queries']:
            print('Not every query aligned, align the remaining queries')
            return
        else:
            I = self.data['num_queries']
            avgDist = 1/I*sum(self.data['distance_distortion'][step_pattern].values())

            return avgDist

    def GetPmax(self, queryID):
        Kq = len(self.data['queries'][queryID][0]['values'])
        Kr = len(self.data['reference'][0]['values'])
        Pmax = np.floor(min(Kq, Kr)/abs(Kq - Kr)) if abs(Kq - Kr) > 0 else Kr
        return Pmax

    def GetGlobalPmax(self):
        Pmaxs = [self.GetPmax(queryID) for queryID in self.data['queriesID']]
        return int(min(Pmaxs))

    def Itakura(self, i, j, N, M, step_pattern):
        patt = re.compile("symmetricP[1-9]+\d*")
        if step_pattern == "symmetricP05":
            p = 1/2
        elif patt.match(step_pattern):
            p = int(step_pattern[step_pattern.index('P')+1:])
        else: return True

        inDomain = (i >= np.floor(j*p/(p+1))) and (i <= np.ceil(j*(p+1)/p)) and \
                    (i <= np.ceil(N+(j-M)*(p/(p+1)))) and (i >= np.floor(N+(j-M)*((p+1)/p)))
        return inDomain

    def ExtremeItakura(self, i, j, N, M, step_pattern):
        case = 0
        patt = re.compile("symmetricP[1-9]+\d*")
        if step_pattern == "symmetricP05":
            p = 1/2
        elif patt.match(step_pattern):
            p = int(step_pattern[step_pattern.index('P')+1:])
        else: return (case, True)

        if (i < np.floor(j*p/(p+1))) or (i < np.floor(N+(j-M)*((p+1)/p))):
            case = 1
            return (case, False)

        inDomain = (i >= np.floor(j*p/(p+1))) and (i <= np.ceil(j*(p+1)/p)) and \
        (i <= np.ceil(N+(j-M)*(p/(p+1)))) and (i >= np.floor(N+(j-M)*((p+1)/p)))

        return (case, inDomain)

# ADD CONDITION ON ALIGNMENT ALREADY PERFORMED
def loadData(n_to_keep=50):
    data_path = "data/ope3_26.pickle"
    with open(data_path, "rb") as infile:
        data = pickle.load(infile)

    opeLen = list()
    pvDataset = list()
    for _id, pvs in data.items():
        opeLen.append((len(pvs[0]['values']), _id))
        pvList = list()
        for pv_dict in pvs:
            pvList.append(pv_dict['name'])
        pvDataset.append(pvList)

    medLen = np.median([l for l, _id in opeLen])

    # Select the N=50 closest to the median bacthes
    # center around the median
    centered = [(abs(l-medLen), _id) for l, _id in opeLen]
    selected = sorted(centered)[:n_to_keep]

    med_id = selected[0][1]  # 5153

    # pop batches without all pvs
    ids = list(data.keys())
    for _id in ids:
        k = len(data[_id])
        if k != 99:
            data.pop(_id)

    allIDs = list(data.keys())
    for _id in allIDs:
        if _id not in [x[1] for x in selected]:
            _ = data.pop(_id)

    data['reference'] = med_id

    return data
