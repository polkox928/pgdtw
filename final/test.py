#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 28 16:39:46 2018

@author: paolograniero
"""
# =============================================================================
# import libdtw as lib
# import numpy as np
# 
# ref = np.array([1,3,5,3,3,3,4,7,5,6]).reshape(-1,1)
# query = np.array([1,1.5,2,1,3,4,2]).reshape(-1,1)
# step_pattern = 'symmetricP1'
# 
# d = lib.dtw()
# 
# distMatrix = d.comp_dist_matrix(ref, query, "euclidean", 1)
# 
# print(distMatrix)
# 
# acc = d.CompAccDistmatrix(distMatrix, step_pattern)
# 
# print(acc)
# print("Dist", acc[-1,-1])
# N,M = acc.shape
# warp = d.GetWarpingPath(acc, step_pattern, N, M)
# print(warp)
# 
# res = d.dtw(ref, query, step_pattern, open_ended = True)
# 
# print(res)
# =============================================================================

import libdtw as lib
import pickle
import matplotlib.pyplot as plt
import numpy as np

data_path = "data/ope3_26.pickle"
with open(data_path, "rb") as infile:
    data = pickle.load(infile)

opeLen = list()
pvDataset = list()
for _id, pvs in data.items():
    opeLen.append((len(pvs[0]['values']), _id))
    pvList = list()
    for pv in pvs:
        pvList.append(pv['name'])
    pvDataset.append(pvList)
#plt.hist([l for l, _id in opeLen], bins=50)
#plt.show()

medLen = np.median([l for l, _id in opeLen])

# Select the N=50 closest to the median bacthes
# center around the median
centered = [(abs(l-medLen), _id) for l, _id in opeLen]
selected = sorted(centered)[:50]
#plt.hist([l for l, _id in opeLen if _id in [ID for d, ID in selected]], bins = 20)
#plt.show()
med_id = selected[0][1] #5153
print(med_id)
# pop batches without all pvs
IDs = list(data.keys())
for _id in IDs:
    L = len(data[_id])
    if L != 99:
        data.pop(_id)
print(len(data))


allIDs = list(data.keys())
for _id in allIDs:
    if _id not in [x[1] for x in selected]:
        _ = data.pop(_id)
print(len(data))
print(data.keys())
IDs = list(data.keys())

data['reference'] = med_id

d = lib.Dtw(data)
ref = d.convert_to_mvts(d.data['reference'])
query = d.convert_to_mvts(d.data['queries']['5110'])

step_pattern = "symmetric2"
res = d.dtw(ref, query, open_ended=False, step_pattern=step_pattern, n_jobs = 1)
warp = res['warping']

print(res['DTW_distance'])
print(res['warping'][-1])
#%%
#fig=plt.figure(figsize=(12, 5))
#fig.suptitle(step_pattern)
#fig.add_subplot(1,2,1)
#d.distance_cost_plot(d.comp_dist_matrix(ref, query))
#plt.plot([x[1] for x in warp], [x[0] for x in warp])
#plt.ylim(0, ref.shape[0])
#plt.xlim(0, query.shape[0])
#plt.title('Distance Matrix')
##plt.show()
#
#fig.add_subplot(1,2,2)
#d.distance_cost_plot(d.comp_acc_dist_matrix(d.comp_dist_matrix(ref, query), step_pattern=step_pattern))
#plt.plot([x[1] for x in warp], [x[0] for x in warp])
#plt.ylim(0, ref.shape[0])
#plt.xlim(0, query.shape[0])
#plt.title('Accumulated Distance')
#plt.show()
#%%

with open("dtwObj50stepPattern.pickle", 'rb') as f:
    D = pickle.load(f)

D.data['warpings_per_step_pattern'].keys()

D.plot_warped_curves('5300', ['ba_FC3BnZ'], 'symmetricP05', True)

#fig = plt.figure(figsize=(12,8))
#query_id = '5300'
#pv_name = ['ba_ErguU']
#step_pattern = 'symmetricP2'
#warping = D.data['warpings_per_step_pattern'][step_pattern][query_id]
#query_warping = [x[1] for x in warping]
#ref_warping = [x[0] for x in warping]
#query = D.data['queries'][query_id]
#ref = D.data['reference']
#
#query_values = list(filter(lambda x: x['name']==pv_name, query))[0]['values']
#warped_query = D.warp_pv(query_values, query_warping)
#ref_values = list(filter(lambda x: x['name']==pv_name, ref))[0]['values']
#warped_ref = D.warp_pv(ref_values, ref_warping)
#
#plt.plot(warped_query, color = 'b')
#plt.plot(warped_ref, color = 'orange')
#plt.xlim((0, len(warping)))
#plt.ylim((0,max(max(query_values), max(ref_values))))
#plt.show()