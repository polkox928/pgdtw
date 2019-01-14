# -*- coding: utf-8 -*-

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
plt.hist([l for l, _id in opeLen], bins=50)
plt.show()

medLen = np.median([l for l, _id in opeLen])

# Select the N=50 closest to the median bacthes
# center around the median
centered = [(abs(l-medLen), _id) for l, _id in opeLen]
selected = sorted(centered)[:50]
plt.hist([l for l, _id in opeLen if _id in [ID for d, ID in selected]], bins = 20)
plt.show()
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

data['reference'] = '5153'

d = lib.dtw(data)
ref = d.ConvertToMVTS(d.data['reference'])
query = d.ConvertToMVTS(d.data['queries']['5341'])

step_pattern = "symmetric2"
res = d.DTW(ref, query, open_ended=False, step_pattern=step_pattern)
warp = res['warping']

print(res['DTW_distance'])
print(res['warping'][-1])

fig=plt.figure(figsize=(12, 5))
fig.add_subplot(1,2,1)
d.DistanceCostPlot(d.CompDistMatrix(ref, query))
plt.plot([x[1] for x in warp], [x[0] for x in warp])
plt.ylim(0, ref.shape[0])
plt.xlim(0, query.shape[0])
plt.title('Distance Matrix')
#plt.show()

fig.add_subplot(1,2,2)
d.DistanceCostPlot(d.CompAccDistMatrix(d.CompDistMatrix(ref, query), step_pattern=step_pattern))
plt.plot([x[1] for x in warp], [x[0] for x in warp])
plt.ylim(0, ref.shape[0])
plt.xlim(0, query.shape[0])
plt.title('Accumulated Distance')
plt.show()
