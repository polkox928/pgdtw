"""
dtw class and load_data function
"""
from collections import defaultdict
import re
import pickle
import multiprocessing
from copy import copy
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from scipy.spatial.distance import euclidean
#import matplotlib.pyplot as plt
#import matplotlib
from joblib import Parallel, delayed
import pandas as pd
from tqdm import tqdm

def load_data(n_to_keep=50):
    """
    Load data of operation 3.26, only the n_to_keep batches with duration closer to the median one
    are selected
    """
    data_path = "data/ope3_26.pickle"
    with open(data_path, "rb") as infile:
        data = pickle.load(infile)

    operation_length = list()
    pv_dataset = list()
    for _id, pvs in data.items():
        operation_length.append((len(pvs[0]['values']), _id))
        pv_list = list()
        for pv_dict in pvs:
            pv_list.append(pv_dict['name'])
        pv_dataset.append(pv_list)

    median_len = np.median([l for l, _id in operation_length])

    # Select the ref_len=50 closest to the median bacthes
    # center around the median
    centered = [(abs(l-median_len), _id) for l, _id in operation_length]
    selected = sorted(centered)[:n_to_keep]

    med_id = selected[0][1]  # 5153

    # pop batches without all pvs
    # ids = list(data.keys())
    # for _id in ids:
    #     k = len(data[_id])
    #     if k != 99:
    #         data.pop(_id)

    all_ids = list(data.keys())
    for _id in all_ids:
        if _id not in [x[1] for x in selected]:
            _ = data.pop(_id)

    data['reference'] = med_id

    return data

class Dtw:
    """
    Everything related to dtw and experimentation
    """

    def __init__(self, json_obj=False):
        """
        Initialization of the class.
        json_obj: contains the data in the usual format
        """
        if not json_obj:
            pass
        else:
            self.convert_data_from_json(json_obj)
            #self.scale_params = self.get_scaling_parameters()
            self.remove_const_feats()
            self.reset_weights()

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

        self.data = {"ref_id": ref_id,
                     "reference": reference,
                     "queries": queries,
                     "num_queries": len(queries),
                     "warpings": dict(),
                     "distances": dict(),
                     'warp_dist': dict(),
                     "queriesID": list(queries.keys()),
                     "time_distortion": defaultdict(dict),
                     "distance_distortion": defaultdict(dict),
                     'warpings_per_step_pattern': defaultdict(dict),
                     'feat_weights': 1.0}

        self.data_open_ended = {"ref_id": ref_id,
                                "reference": reference,
                                "queries": defaultdict(list),
                                'warp_dist': dict()}
        scale_params = dict()

        for pv_dict in self.data['reference']:
            pv_name = pv_dict['name']
            pv_min = min(pv_dict['values'])
            pv_max = max(pv_dict['values'])
            scale_params[pv_name] = (pv_min, pv_max)

        self.scale_params = scale_params


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

        initial_queries = list(self.data['queries'].keys())
        print('Number of queries before filtering: %d'%len(initial_queries))

        self.data['reference'] = list(filter(lambda x: x['name'] not in const_feats, self.data['reference']))

        for _id in initial_queries:
            self.data['queries'][_id] = list(filter(lambda x: x['name'] not in const_feats, self.data['queries'][_id]))
            if len(self.data['queries'][_id]) != len(self.data['reference']):
                _ = self.data['queries'].pop(_id)
        print('Number of queries after filtering: %d'%len(self.data['queries']))

        self.data['num_queries'] = len(self.data['queries'])
        self.data['queriesID'] = list(self.data['queries'].keys())

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
            pv_min, pv_max = self.scale_params[pv_name]
            scaled_pv_values = (np.array(pv_values)-pv_min)/(pv_max-pv_min)
        return scaled_pv_values

    def convert_to_mvts(self, batch):     # mvts = Multi Variate Time Series
        """
        Takes one batch in the usual form (list of one dictionary per PV) and transforms
        it to a numpy array to perform calculations faster
        """
        k = len(batch[0]['values'])  # Length of a batch (number of data points per single PV)
        num_feat = len(batch)  # Number of PVs

        mvts = np.zeros((k, num_feat))

        for (i, pv_dict) in zip(np.arange(num_feat), batch):
            mvts[:, i] = self.scale_pv(pv_dict['name'], pv_dict['values'], "group")

        return mvts

    def comp_dist_matrix(self, reference_ts, query_ts, n_jobs=1):
        """
        Computes the distance matrix with ref_len (length of the reference) number of rows and
        query_len (length of the query) number of columns (OK with convention on indices in dtw)
        with dist_measure as local distance measure

        reference_ts: mvts representation of reference batch
        query_ts: mvts representation of query batch

        n_jobs: number of jobs for pairwise_distances function. It could cause problems on windows
        """
        _, d_1 = reference_ts.shape
        _, d_2 = query_ts.shape

        if d_1 != d_2:
            print("Number of features not coherent between reference ({0}) and query ({1})"
                  .format(d_1, d_2))
            return None

        distance_matrix = pairwise_distances(
            X=reference_ts, Y=query_ts, metric=euclidean, n_jobs=n_jobs, w=self.data['feat_weights']
            )

        return distance_matrix

    def comp_acc_dist_matrix(self, distance_matrix, step_pattern='symmetricP05', open_ended=False):
        """
        Computes the accumulated distance matrix starting from the distance_matrix according to the
        step_pattern indicated
        distance_matrix: cross distance matrix
        step_pattern: string indicating the step pattern to be used. Can be symmetric1/2,
        symmetricP05 or symmetricPX, with X any positive integer
        """
        ref_len, query_len = distance_matrix.shape
        acc_dist_matrix = np.empty((ref_len, query_len))
        if not open_ended:
            for i in np.arange(ref_len):
                for j in np.arange(query_len):
                    acc_dist_matrix[i, j] = self.comp_acc_element(
                        i, j, acc_dist_matrix, distance_matrix, step_pattern)\
                        if self.itakura(i, j, ref_len, query_len, step_pattern) else np.inf

        else:
            for i in np.arange(ref_len):
                for j in np.arange(query_len):
                    acc_dist_matrix[i, j] = self.comp_acc_element(
                        i, j, acc_dist_matrix, distance_matrix, step_pattern)
        return acc_dist_matrix

    def comp_acc_element(self, i, j, acc_dist_matrix, distance_matrix, step_pattern):
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

            p_1 = acc_dist_matrix[i-1, j-3] + 2 * distance_matrix[i, j-2] + distance_matrix[i, j-1]\
                + distance_matrix[i, j] if (i-1 >= 0 and j-3 >= 0) else np.inf
            p_2 = acc_dist_matrix[i-1, j-2] + 2 * distance_matrix[i, j-1] + \
                distance_matrix[i, j] if (i-1 >= 0 and j-2 >= 0) else np.inf
            p_3 = acc_dist_matrix[i-1, j-1] + 2 * \
                distance_matrix[i, j] if (i-1 >= 0 and j-1 >= 0) else np.inf
            p_4 = acc_dist_matrix[i-2, j-1] + 2 * distance_matrix[i-1, j] + \
                distance_matrix[i, j] if (i-2 >= 0 and j-1 >= 0) else np.inf
            p_5 = acc_dist_matrix[i-3, j-1] + 2 * distance_matrix[i-2, j] + distance_matrix[i-1, j]\
                + distance_matrix[i, j] if (i-3 >= 0 and j-1 >= 0) else np.inf

            return min(p_1, p_2, p_3, p_4, p_5)  # /sum(acc_dist_matrix.shape)

        if step_pattern == "symmetric1":
            p_1 = acc_dist_matrix[i, j-1] + distance_matrix[i, j] if (j-1 >= 0) else np.inf
            p_2 = acc_dist_matrix[i-1, j-1] + distance_matrix[i, j]\
                if (i-1 >= 0 and j-1 >= 0) else np.inf
            p_3 = acc_dist_matrix[i-1, j] + distance_matrix[i, j] if (i-1 >= 0) else np.inf

            return min(p_1, p_2, p_3)

        if step_pattern == "symmetric2":
            p_1 = acc_dist_matrix[i, j-1] + distance_matrix[i, j] if (j-1 >= 0) else np.inf
            p_2 = acc_dist_matrix[i-1, j-1] + 2 * \
                distance_matrix[i, j] if (i-1 >= 0 and j-1 >= 0) else np.inf
            p_3 = acc_dist_matrix[i-1, j] + distance_matrix[i, j] if (i-1 >= 0) else np.inf

            return min(p_1, p_2, p_3)  # /sum(acc_dist_matrix.shape)

        patt = re.compile("symmetricP[1-9]+\d*")
        if patt.match(step_pattern):
            p = int(step_pattern[10:])
            p_1 = acc_dist_matrix[i-p, j-(p+1)] + 2*sum([distance_matrix[i-p, j-(p+1)]\
                                             for p in np.arange(0, p)]) + distance_matrix[i, j]\
                                                        if (i-p >= 0 and j-(p+1) >= 0) else np.inf
            p_2 = acc_dist_matrix[i-1, j-1] + \
                2 * distance_matrix[i, j] if (i-1 >= 0 and j-1 >= 0) else np.inf
            p_3 = acc_dist_matrix[i-(p+1), j-p] + 2*sum([distance_matrix[i-(p+1), j-p]\
                                             for p in np.arange(0, p)]) + distance_matrix[i, j] \
                                                                    if (i-(p+1) >= 0 and j-p >= 0) \
                                                                                        else np.inf

            return min(p_1, p_2, p_3)  # /sum(acc_dist_matrix.shape)

    def get_warping_path(self, acc_dist_matrix, step_pattern, ref_len, query_len):
        """
        Computes the warping path on the acc_dist_matrix induced by step_pattern starting from
        the (ref_len,query_len) point (this in order to use the method in both open_ended and global
        alignment)
        Return the warping path (list of tuples) in ascending order
        """
        # ref_len, query_len = acc_dist_matrix.shape
        warping_path = list()

        if step_pattern == "symmetric1" or step_pattern == "symmetric2":
            i = ref_len-1
            j = query_len-1
            while i != 0 or j != 0:
                warping_path.append((i, j))
                candidates = list()
                if i > 0:
                    candidates.append((acc_dist_matrix[i-1, j], (i-1, j)))
                if j > 0:
                    candidates.append((acc_dist_matrix[i, j-1], (i, j-1)))
                if len(candidates) == 2:
                    candidates.append((acc_dist_matrix[i-1, j-1], (i-1, j-1)))

                next_step = min(candidates)[1]
                i, j = next_step
            warping_path.append((0, 0))

            return warping_path[::-1]

        elif step_pattern == "symmetricP05":
            # maxWarp = 2
            # minDiag = 1
            i = ref_len-1
            j = query_len-1

            if np.isnan(acc_dist_matrix[i, j]):
                print("Invalid value for P, \
                      a global alignment is not possible with this local constraint")
                return
            h_step = 0  # horizontal step
            v_step = 0  # vertical step
            d_step = 0  # diagonal step

            while i != 0 or j != 0:
                warping_path.append((i, j))
                candidates = list()

                if h_step > 0:
                    if h_step == 1:
                        if j > 0:
                            candidates.append((acc_dist_matrix[i, j-1], (i, j-1)))
                        if j > 0 and i > 0:
                            candidates.append((acc_dist_matrix[i-1, j-1], (i-1, j-1)))
                    elif h_step == 2:
                        if j > 0 and i > 0:
                            candidates.append((acc_dist_matrix[i-1, j-1], (i-1, j-1)))

                elif v_step > 0:
                    if v_step == 1:
                        if i > 0:
                            candidates.append((acc_dist_matrix[i-1, j], (i-1, j)))
                        if j > 0 and i > 0:
                            candidates.append((acc_dist_matrix[i-1, j-1], (i-1, j-1)))
                    elif v_step == 2:
                        if j > 0 and i > 0:
                            candidates.append((acc_dist_matrix[i-1, j-1], (i-1, j-1)))

                else:
                    if j > 0:
                        candidates.append((acc_dist_matrix[i, j-1], (i, j-1)))
                    if i > 0:
                        candidates.append((acc_dist_matrix[i-1, j], (i-1, j)))
                    if j > 0 and i > 0:
                        candidates.append((acc_dist_matrix[i-1, j-1], (i-1, j-1)))

                next_step = min(candidates)[1]
                v = next_step[0] < i
                h = next_step[1] < j
                d = v and h

                if d:
                    v_step = 0
                    h_step = 0
                elif v:
                    v_step += 1
                elif h:
                    h_step += 1

                i, j = next_step

            warping_path.append((0, 0))

            return warping_path[::-1]

        else:
            patt = re.compile("symmetricP[1-9]+\d*")
            if patt.match(step_pattern):

                min_diag_steps = int(step_pattern[10:])

                warp_step = 0
                d_step = 0
                i = ref_len-1
                j = query_len-1

                if np.isinf(acc_dist_matrix[i, j]):
                    print("Invalid value for P, \
                          a global alignment is not possible with this local constraint")
                    return

                while i != 0 and j != 0:
                    warping_path.append((i, j))
                    candidates = list()
                    if warp_step > 0:
                        candidates.append((acc_dist_matrix[i-1, j-1], (i-1, j-1)))
                    else:
                        if j > 0:
                            candidates.append((acc_dist_matrix[i, j-1], (i, j-1)))
                        if i > 0:
                            candidates.append((acc_dist_matrix[i-1, j], (i-1, j)))
                        if len(candidates) == 2:
                            candidates.append((acc_dist_matrix[i-1, j-1], (i-1, j-1)))

                    next_step = min(candidates)[1]
                    v = next_step[0] < i
                    h = next_step[1] < j
                    d = v and h

                    if d:
                        d_step += 1
                        if d_step == min_diag_steps:
                            d_step = 0
                            warp_step = 0
                        elif d_step < min_diag_steps and warp_step > 0:
                            pass
                        elif d_step < min_diag_steps and warp_step == 0:
                            d_step = 0
                    else:
                        warp_step += 1

                    i, j = next_step

                warping_path.append((0, 0))

                return warping_path[::-1]

            else:
                print("Invalid step-pattern")

    def call_dtw(self, query_id, step_pattern="symmetricP05",
                 n_jobs=1, open_ended=False, get_results=False, length = 0, all_sub_seq=False):
        """
        Calls the dtw method on the data stored in the .data attribute (needs only the query_id in \
        addition to standard parameters)
        get_results if True returns the distance and the warping calculated; if False, \
        only the .data attribute is updated
        """
        if not open_ended:
            if step_pattern in self.data['warpings_per_step_pattern']:
                if query_id in self.data['warpings_per_step_pattern'][step_pattern]:
                    return

            reference_ts = self.convert_to_mvts(self.data['reference'])
            query_ts = self.convert_to_mvts(self.data['queries'][query_id])

            result = self.dtw(reference_ts, query_ts, step_pattern, n_jobs, open_ended)

            self.data["warpings"][query_id] = result["warping"]
            self.data["distances"][query_id] = result["DTW_distance"]
            self.data['warp_dist'][query_id]=list()
            for (i,j) in result["warping"]:
                self.data['warp_dist'][query_id].append((i, j, result['acc_matrix'][i, j]/(i+j+2)))

            self.data['time_distortion'][step_pattern][query_id] = \
                self.time_distortion(result['warping'])
            self.data['distance_distortion'][step_pattern][query_id] = result["DTW_distance"]
            self.data['warpings_per_step_pattern'][step_pattern][query_id] = result['warping']

            if get_results:
                return result

        if open_ended:
            # ADD ALL SUB SEQUENCE OPTION#############################
            if all_sub_seq:
                reference_ts = self.convert_to_mvts(self.data['reference'])
                query_ts = self.convert_to_mvts(self.data['queries'][query_id])

                result = self.dtw(reference_ts, query_ts, step_pattern, n_jobs, open_ended=False)

                acc_dist_matrix = result['acc_matrix']
                N, M = acc_dist_matrix.shape
                self.data_open_ended['warp_dist'][query_id] = list()
                for j in np.arange(M):
                    candidate = acc_dist_matrix[:, j]
                    candidate /= np.arange(2, N+2) + j
                    i_min, dtw_dist = np.argmin(candidate), min(candidate)

                    self.data_open_ended['warp_dist'][query_id].append((i_min, j, dtw_dist))
                return

            if not length:
                print("Length cannot be 0")
                return

            if not self.check_open_ended(query_id, length, step_pattern):

                query_ts = self.convert_to_mvts(self.online_query(query_id, length))
                reference_ts = self.convert_to_mvts(self.data['reference'])

                result = self.dtw(reference_ts, query_ts, step_pattern, n_jobs, open_ended)

                data_point = {'length': length,
                              'DTW_distance': result['DTW_distance'],
                              'warping':result['warping'],
                              'step_pattern': step_pattern}
                self.data_open_ended['queries'][query_id].append(data_point)

            if get_results:
                return list(filter(lambda x: x['step_pattern']==step_pattern and x['length']==length, self.data_open_ended['queries'][query_id]))[0]


    def dtw(self, reference_ts, query_ts, step_pattern="symmetricP05",
            n_jobs=1, open_ended=False):
        """
        Compute alignment betwwen reference_ts and query_ts (already in mvts form).
        Separate from call_dtw() for testing purposes
        """
        # Check for coherence of local constraint and global alignment
        # (in case a PX local constraint is used)
        if not open_ended:
            patt = re.compile("symmetricP[1-9]+\d*")
            if patt.match(step_pattern):
                p = int(step_pattern[step_pattern.index("P")+1:])
                ref_len, query_len = len(reference_ts), len(query_ts)
                p_max = np.floor(min(ref_len, query_len)/np.abs(ref_len-query_len)) \
                    if np.abs(ref_len-query_len) > 0 else np.inf
                if p > p_max:
                    print("Invalid value for P, \
                                  a global alignment is not possible with this local constraint")
                    return
            else:
                pass

        distance_matrix = self.comp_dist_matrix(reference_ts, query_ts, n_jobs)

        acc_dist_matrix = self.comp_acc_dist_matrix(distance_matrix, step_pattern, open_ended)

        ref_len, query_len = acc_dist_matrix.shape
        # In case of open-ended version
        # correctly identifies the starting point on the reference batch for warping
        if open_ended:
            ref_len = self.get_ref_prefix_length(acc_dist_matrix)

        warping = self.get_warping_path(acc_dist_matrix, step_pattern, ref_len, query_len)

        dtw_dist = acc_dist_matrix[ref_len-1, query_len-1] / (ref_len+query_len)

        return {"warping": warping,
                "DTW_distance": dtw_dist,
                'acc_matrix': acc_dist_matrix}

    def get_ref_prefix_length(self, acc_dist_matrix):
        """
        Computes the length of the reference prefix in case of open-ended alignment
        """
        N, M = acc_dist_matrix.shape
        last_column = acc_dist_matrix[:, -1]/np.arange(1, N+1)

        ref_prefix_len = np.argmin(last_column) + 1
        return ref_prefix_len

    def distance_cost_plot(self, distance_matrix):
        """
        Draws a heatmap of distance_matrix, nan values are colored in green
        """
        cmap = matplotlib.cm.inferno
        cmap.set_bad('green', .3)
        masked_array = np.ma.array(distance_matrix, mask=np.isnan(distance_matrix))
        img = plt.imshow(masked_array, interpolation='nearest', cmap=cmap)

        plt.gca().invert_yaxis()
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.grid()
        plt.colorbar()

    def time_distortion(self, warping_path):
        """
        Computes the time distortion caused by warping_path
        """
        T = len(warping_path)
        f_q = [w[1] for w in warping_path]
        f_r = [w[0] for w in warping_path]

        t_d = [(f_r[t+1] - f_r[t])*(f_q[t+1] - f_q[t]) == 0 for t in np.arange(T-1)]

        return sum(t_d)

    def avg_time_distortion(self, step_pattern):
        """
        Computes the average time distortion relative to a certain step pattern
        """
        if len(self.data['time_distortion'][step_pattern]) != self.data['num_queries']:
            print('Not every query aligned, align the remaining queries')
            return
        else:
            I = self.data['num_queries']
            avg_td = sum(self.data['time_distortion'][step_pattern].values())/I

            return avg_td

    def avg_distance(self, step_pattern):
        """
        Computes average
        """
        if len(self.data['distance_distortion'][step_pattern]) != self.data['num_queries']:
            print('Not every query aligned, align the remaining queries')
            return
        else:
            I = self.data['num_queries']
            avg_dist = sum(self.data['distance_distortion'][step_pattern].values())/I

            return avg_dist

    def get_p_max(self, query_id):
        """
        Computes the maximum value of P for the selected query batch
        """
        k_q = len(self.data['queries'][query_id][0]['values'])
        k_r = len(self.data['reference'][0]['values'])
        p_max = np.floor(min(k_q, k_r)/abs(k_q - k_r)) if abs(k_q - k_r) > 0 else k_r
        return p_max

    def get_global_p_max(self):
        """
        Computes the maximum value of P for the data set under consideration
        """
        p_maxs = [self.get_p_max(query_id) for query_id in self.data['queriesID']]
        return int(min(p_maxs))

    def itakura(self, i, j, ref_len, query_len, step_pattern):
        """
        Induced Itakura global constraint for GLOBAL ALIGNMENT
        """
        patt = re.compile("symmetricP[1-9]+\d*")
        if step_pattern == "symmetricP05":
            p = 1/2
        elif patt.match(step_pattern):
            p = int(step_pattern[step_pattern.index('P')+1:])
        else:
            return True

        in_domain = (i >= np.floor(j*p/(p+1))) and \
                    (i <= np.ceil(j*(p+1)/p)) and \
                    (i <= np.ceil(ref_len+(j-query_len)*(p/(p+1)))) and \
                    (i >= np.floor(ref_len+(j-query_len)*((p+1)/p)))
        return in_domain

    def extreme_itakura(self, i, j, ref_len, query_len, step_pattern):
        """
        Alternative implementation of itakura method
        """
        case = 0
        patt = re.compile("symmetricP[1-9]+\d*")
        if step_pattern == "symmetricP05":
            p = 1/2
        elif patt.match(step_pattern):
            p = int(step_pattern[step_pattern.index('P')+1:])
        else:
            return (case, True)

        if (i < np.floor(j*p/(p+1))) or (i < np.floor(ref_len+(j-query_len)*((p+1)/p))):
            case = 1
            return (case, False)

        in_domain = (i >= np.floor(j*p/(p+1))) and \
                    (i <= np.ceil(j*(p+1)/p)) and \
                    (i <= np.ceil(ref_len+(j-query_len)*(p/(p+1)))) and \
                    (i >= np.floor(ref_len+(j-query_len)*((p+1)/p)))

        return (case, in_domain)

    def reset_weights(self):
        """
        Reset the variables' weights to 1
        """
        n_feat = len(self.data['reference'])
        weigths = np.ones(n_feat)
        self.data['feat_weights'] = weigths

    def compute_mld(self, distance_matrix, warping_path):
        """
        Compute the MLDs coefficients (mean local distance) of a certain distance_matrix relative
        to warping_path
        """
        k = len(warping_path)
        on_path = [distance_matrix[i, j] for i, j in warping_path]
        on_path_mld = np.mean(on_path)
        off_path_mld = (sum(sum(distance_matrix)) - sum(on_path))\
            / (np.product(distance_matrix.shape)-k)

        return {'onpath': on_path_mld,
                'offpath': off_path_mld}

    def extract_single_feat(self, feat_idx, query_id):
        """
        Accessory method for selecting single features from the dataset
        """
        pv_name = self.data['reference'][feat_idx]['name']
        reference_ts = np.array(self.scale_pv(pv_name, self.data['reference'][feat_idx]['values'], mode = 'group')).reshape(-1, 1)
        query_ts = np.array(self.scale_pv(pv_name, self.data['queries'][query_id][feat_idx]['values'], mode = 'group')).reshape(-1, 1)

        return {'reference': reference_ts,
                'query': query_ts}

    def weight_optimization_single_batch(self, query_id, step_pattern, n_jobs):
        """
        Optimization step regarding a single batch
        """
        reference_ts = self.convert_to_mvts(self.data['reference'])
        query_ts = self.convert_to_mvts(self.data['queries'][query_id])
        res = self.dtw(reference_ts, query_ts, step_pattern=step_pattern, n_jobs=-1)
        warping = res['warping']
        tot_feats = len(self.data['reference'])
        inputs = np.arange(tot_feats)

        num_cores = min(n_jobs, multiprocessing.cpu_count() - 1)

        def process_feats(feat_idx):
            """
            Computes mld->weight for single feature
            """
            single_feats = self.extract_single_feat(feat_idx, query_id)
            reference = single_feats['reference']
            query = single_feats['query']
            local_distance_matrix = self.comp_dist_matrix(reference, query, num_cores)

            mld = self.compute_mld(local_distance_matrix, warping)

            weight = mld['offpath']/mld['onpath'] if mld['onpath'] > 1e-6 else 1.0
            return weight

        weights = Parallel(n_jobs=1)\
                                    (delayed(process_feats)(feat_idx) for feat_idx in inputs)

        return weights

    def weight_optimization_step(self, step_pattern='symmetric2', update=False, n_jobs=1):
        """
        Single iteration of the optimization algorithm, considering all batches in the instance
        """
        tot_feats = len(self.data['reference'])
        num_queries = self.data['num_queries']
        w_matrix = np.empty((num_queries, tot_feats))

        for c, query_id in tqdm(zip(np.arange(num_queries), self.data['queriesID']), desc='Batch Processing', total=num_queries, leave=False):
            #print('\rBatch %3d/%3d'%(c+1, num_queries), end = '')
            w_matrix[c, ] = self.weight_optimization_single_batch(query_id, step_pattern, n_jobs)

        updated_weights = np.mean(w_matrix, axis=0)
        updated_weights = updated_weights/sum(updated_weights) * tot_feats

        if update:
            self.data['feat_weights'] = updated_weights

        return updated_weights

    def optimize_weights(self, step_pattern='symmetric2', convergence_threshold=0.01, n_steps=10,\
                                                                             file_path=0, n_jobs=1):
        """
        Implements the algorithm for the optimization of the weights
        """
        current_weights = self.data['feat_weights']
        old_weights = self.data['feat_weights']
        conv_val = 1
        step = 0

        while conv_val > convergence_threshold and step < n_steps:
            updated_weights = self.weight_optimization_step(step_pattern, update=True,\
                                                            n_jobs=n_jobs)
            loop_conv_val = np.linalg.norm(updated_weights - old_weights, ord=2)\
                            / np.linalg.norm(old_weights, ord=2)
            if loop_conv_val <= convergence_threshold*0.1:
                print('Algorithm inside a loop')
                break
            conv_val = np.linalg.norm(updated_weights - current_weights, ord=2)\
                / np.linalg.norm(current_weights, ord=2)
            old_weights = current_weights
            current_weights = updated_weights
            step += 1
            print('\nConvergence value: %0.3f\nStep: %d\n' % (conv_val, step))
            print(current_weights, '\n')
            if file_path:
                with open(file_path, 'wb') as f:
                    pickle.dump(updated_weights, f, protocol=pickle.HIGHEST_PROTOCOL)

        self.data['feat_weights'] = updated_weights

    def get_weight_variables(self):
        """
        Returns a dictionary with the weight for each variable
        """
        var_names = [pv['name'] for pv in self.data['reference']]
        var_weight = {var: weight for var, weight in zip(var_names, self.data['feat_weights'])}
        return var_weight

    def plot_weights(self, n=25, figsize=(15, 8)):
        """
        Horizontal bar chart with variables' weights sorted by magnitude
        """
        plt.rcdefaults()
        fig, ax = plt.subplots(figsize=figsize)

        var_names = sorted(list(self.get_weight_variables().items()),
                      key=lambda x: x[1], reverse=True)[:n]
        names = [v[0] for v in var_names]
        y_pos = np.arange(len(names))
        weights = [v[1] for v in var_names]

        ax.barh(y_pos, weights, align='center', color='#d90000')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names)
        ax.invert_yaxis()  # labels read top-to-bottom
        ax.set_xlabel('Weights')
        ax.set_title('Variables\' weights')

        fig.tight_layout()
        plt.show()

    def plot_by_name(self, _id, pv_name):
        """
        Plots one pv relative to a batch with ID equal to _id, according to its name pv_name
        """
        pv_list = [pv['name'] for pv in self.data['reference']]

        if _id != self.data['ref_id']:
            pv_idx = pv_list.index(pv_name)
            plt.plot(self.data['queries'][_id][pv_idx]['values'])
            plt.show()
        elif _id == self.data['ref_id']:
            pv_idx = pv_list.index(pv_name)
            plt.plot(self.data['reference'][pv_idx]['values'])
            plt.show()
        else: print('Batch ID not found')

    def do_warp(self, ref_values, query_values, warping_path, symmetric=True):
        """
        Performs warping of reference and query values.
        Symmetric: reference and query warped to common time axis
        Asymmetric: query warped to reference time axis averaging warped elements
        """
        query_warping = [x[1] for x in warping_path]
        ref_warping = [x[0] for x in warping_path]

        if symmetric:
            warped_query = [query_values[j] for j in query_warping]
            warped_ref = [ref_values[i] for i in ref_warping]


        if not symmetric:
            warped_ref = ref_values
            warped_query = list()
            N = len(ref_values)
            for i in range(N):
                warp_idx = [x[1] for x in filter(lambda x: x[0] == i, warping_path)]
                to_warp = [query_values[j] for j in warp_idx]

                warped_query.append(np.mean(to_warp))

        return [warped_ref, warped_query]

    def plot_warped_curves(self, query_id, pv_list, step_pattern, symmetric=False):
        """
        Plot warping curves for all pvs in pv_list, for both reference and query batch
        """
        if isinstance(pv_list, str):
            pv_list = [pv_list]

        warping = self.data['warpings_per_step_pattern'][step_pattern][query_id]
        query = self.data['queries'][query_id]
        ref = self.data['reference']

        fig = plt.figure(figsize=(12, 8))

        for pv_name in pv_list:
            query_values = list(filter(lambda x: x['name'] == pv_name, query))[0]['values']
            ref_values = list(filter(lambda x: x['name'] == pv_name, ref))[0]['values']
            warped_ref, warped_query = self.do_warp(ref_values, query_values, warping, symmetric)

            plt.plot(warped_query, color='b', label="Query")
            plt.plot(warped_ref, color='orange', label='Reference')

        plt.legend()
        plt.title('Step pattern: %s'%step_pattern)
        plt.xlim((0, len(warped_ref)))
        plt.ylim((0, max(max(query_values), max(ref_values))+5))
        plt.show()

    def online_scale(self, pv_dict_online):
        pv_name = pv_dict_online['name']
        pv_values = np.array(pv_dict_online['values'])
        pv_min, pv_max = self.scale_params[pv_name]
        scaled_values = (pv_values - pv_min)/(pv_max - pv_min) if pv_max - pv_min > 10-6 else np.full(pv_values.shape, 0.5)

        return scaled_values

    def online_query(self, query_id, length):
        query = self.data['queries'][query_id]

        cut_query = [{'name':query_pv['name'], 'values':self.online_scale({'name':query_pv['name'], 'values':query_pv['values'][:length]})} for query_pv in query]

        return cut_query

    def check_open_ended(self, query_id, length, step_pattern):
        check_id = query_id in self.data_open_ended['queries']
        if check_id:
            check = bool(list(filter(lambda x: x['step_pattern']==step_pattern and x['length']==length, self.data_open_ended['queries'][query_id])))
            return check
        else:
            return False

    def generate_train_set(self, n_rows=100, step_pattern='symmetricP2', n_jobs = 1, seed=42, query_id = False):

        rand_gen = np.random.RandomState(seed)
        data_set = list()
        id_set = list()
        len_set = list()
        if not query_id:
            for i in np.arange(n_rows):
                query_id = rand_gen.choice(self.data['queriesID'])
                id_set.append(query_id)
                len_set.append(rand_gen.randint(low = 1, high = len(self.data['queries'][query_id][0]['values'])))


        else:
                max_len = len(self.data['queries'][query_id][0]['values'])
                id_set = [query_id]*max_len
                len_set = np.arange(1, max_len+1)
        ref_len = len(self.data['reference'][0]['values'])


        if n_jobs != 1:
            def generate_data_point(_id_length):
                query_id, length = _id_length
                data_point = list(filter(lambda x: x['step_pattern']==step_pattern and x['length']==length, self.data_open_ended['queries'][query_id]))
                if data_point:
                    data_point = copy(data_point[0])
                    data_point['ref_prefix'] = data_point['warping'][-1][0]+1
                    _= data_point.pop('warping')
                    data_point['true_length'] = len(self.data['queries'][query_id][0]['values'])
                    data_point['ref_len'] = ref_len
                    data_point['query_id'] = query_id

                else:
                    data_point = copy(self.call_dtw(query_id, step_pattern, open_ended =True, get_results = True, length = length))
                    data_point['ref_prefix'] = data_point['warping'][-1][0]+1
                    _ = data_point.pop('warping')
                    data_point['true_length'] = len(self.data['queries'][query_id][0]['values'])
                    data_point['ref_len'] = ref_len
                    data_point['query_id'] = query_id

                return data_point

            data_set = Parallel(n_jobs = n_jobs, verbose = 5)(delayed(generate_data_point)(_id_length) for _id_length in zip(id_set, len_set))

        else:
            for query_id, length in tqdm(zip(id_set, len_set)):
                data_point = list(filter(lambda x: x['step_pattern']==step_pattern and x['length']==length, self.data_open_ended['queries'][query_id]))
                if data_point:
                    data_point = copy(data_point[0])
                    data_point['ref_prefix'] = data_point['warping'][-1][0]+1
                    _= data_point.pop('warping')
                    data_point['true_length'] = len(self.data['queries'][query_id][0]['values'])
                    data_point['ref_len'] = ref_len
                    data_point['query_id'] = query_id
                    data_set.append(data_point)
                else:
                    data_point = copy(self.call_dtw(query_id, step_pattern, open_ended =True, get_results = True, length = length))
                    data_point['ref_prefix'] = data_point['warping'][-1][0]+1
                    _ = data_point.pop('warping')
                    data_point['true_length'] = len(self.data['queries'][query_id][0]['values'])
                    data_point['ref_len'] = ref_len
                    data_point['query_id'] = query_id
                    data_set.append(data_point)

        return pd.DataFrame(data_set)
