# -*- coding: utf-8 -*-
# %%
import libdtw as lib
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm
import sys
try:
    n_data = int(sys.argv[1])
except:
    n_data = 50

try:
    with open('dtwObj%d.pickle' % n_data, 'rb') as f:
        d = pickle.load(f)
except:
    data = lib.loadData(n_data)
    d = lib.dtw(data)


possible_step_pattern = ['symmetricP05']
possible_P_pattern = ['symmetricP%s' % p for p in np.arange(1, d.GetGlobalPmax()+1)]

possible_step_pattern.extend(possible_P_pattern)
# %%
res = defaultdict(list)

for step_pattern in possible_step_pattern[::-1]:
    print(step_pattern)
    for _id in tqdm(d.data['queriesID']):
        d.CallDTW(_id, step_pattern=step_pattern, n_jobs=1)

    res[step_pattern].append(d.AvgTimeDistortion(step_pattern))
    res[step_pattern].append(d.AvgDistance(step_pattern))
# %%
    # pickle d object
with open('dtwObj%d.pickle' % n_data, 'wb') as f:
    pickle.dump(d, f, protocol=pickle.HIGHEST_PROTOCOL)

# %%
#possible_step_pattern = [possible_step_pattern[0]] + possible_step_pattern[2:]
td = [x[0] for x in res.values()]
range_td = min(td), max(td)

dist = [x[1] for x in res.values()]
range_dist = min(dist), max(dist)

res_scaled = defaultdict(list)
for step_pattern in possible_step_pattern:
    res_scaled[step_pattern] = [(res[step_pattern][0] - range_td[0])/(range_td[1]-range_td[0])]
    res_scaled[step_pattern].append(
        (res[step_pattern][1] - range_dist[0])/(range_dist[1]-range_dist[0]))


distances = [res_scaled[step_pattern][0] for step_pattern in possible_step_pattern]
distortions = [res_scaled[step_pattern][1] for step_pattern in possible_step_pattern]
score = [np.sqrt(x**2 + y**2) for x, y in zip(distances, distortions)]
x = np.arange(1, len(score)+1)
fig = plt.figure(figsize=(12, 5))
fig.add_subplot(1, 2, 1)
plt.plot(x, score)
plt.xticks(x, possible_step_pattern, rotation="vertical")
plt.ylabel("Alignment score")

fig.add_subplot(1, 2, 2)
plt.plot(distances, distortions, '-o')
plt.xlabel("Scaled distance")
plt.ylabel("Scaled distortion")
for x, y, label in zip(distances, distortions, possible_step_pattern):
    plt.annotate(label[label.index('P'):], xy=(x, y))

plt.show()
