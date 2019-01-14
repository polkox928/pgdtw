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
query = d.ConvertToMVTS(d.data['queries']['5158'])
res = d.DTW(ref, query)
warp = res['warping']

plt.plot([x[0] for x in warp], [x[1] for x in warp])
plt.show()
print(res['DTW_distance'])
