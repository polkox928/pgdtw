"""
Select optimal local constraint based on SCORE (time distortion, dtw distance)
specific for the set of batches under examination
"""  # -*- coding: utf-8 -*-
# %%
from collections import defaultdict
import pickle
import sys
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import libdtw as lib

if __name__ == '__main__':
    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"

    try:
        N_DATA = int(sys.argv[1]) if int(sys.argv[1]) >= 2 else 200
    except LookupError as ex:
        N_DATA = 26

    try:
        with open('stepPattern.pickle', 'rb') as f:
            D = pickle.load(f)
    except OSError as ex:
        DATA = lib.load_data(N_DATA)
        D = lib.Dtw(DATA)
        with open('optWeights.pickle', 'rb') as f:
            D_weights = pickle.load(f)
        D.data['feat_weights'] = D_weights

    POSSIBLE_STEP_PATTERNS = ['symmetricP05']
    POSSIBLE_P_PATTERNS = ['symmetricP%s'%p for p in np.arange(1, D.get_global_p_max()+1)]

    POSSIBLE_STEP_PATTERNS.extend(POSSIBLE_P_PATTERNS)

    RES = defaultdict(list)
    for step_pattern in POSSIBLE_STEP_PATTERNS[::-1]:
        print(step_pattern)
        for _id in tqdm(D.data['queriesID']):
            D.call_dtw(_id, step_pattern=step_pattern, n_jobs=1)

        RES[step_pattern].append(D.avg_time_distortion(step_pattern))
        RES[step_pattern].append(D.avg_distance(step_pattern))

        # pickle D object
    with open('stepPattern.pickle', 'wb') as f:
        pickle.dump(D, f, protocol=pickle.HIGHEST_PROTOCOL)

    TD = [x[0] for x in RES.values()]
    RANGE_TD = min(TD), max(TD)

    DIST = [x[1] for x in RES.values()]
    RANGE_DIST = min(DIST), max(DIST)

    RES_SCALED = defaultdict(list)
    for step_pattern in POSSIBLE_STEP_PATTERNS:
        RES_SCALED[step_pattern] = [(RES[step_pattern][0] - RANGE_TD[0])/(RANGE_TD[1]-RANGE_TD[0])]
        RES_SCALED[step_pattern].append((RES[step_pattern][1] - RANGE_DIST[0])/(RANGE_DIST[1]-RANGE_DIST[0]))


    DISTORTIONS = [RES_SCALED[step_pattern][0] for step_pattern in POSSIBLE_STEP_PATTERNS]
    DISTANCES = [RES_SCALED[step_pattern][1] for step_pattern in POSSIBLE_STEP_PATTERNS]
    SCORE = [np.sqrt(x**2 + y**2) for x, y in zip(DISTANCES, DISTORTIONS)]

    X = np.arange(1, len(SCORE)+1)

    FIG = plt.figure(figsize=(12, 5))
    FIG.add_subplot(1, 2, 1)

    plt.plot(X, SCORE, '-o', color = '#d90000')
    plt.xticks(X, POSSIBLE_STEP_PATTERNS, rotation="vertical")
    plt.ylabel("Alignment SCORE")

    FIG.add_subplot(1, 2, 2)
    plt.plot(DISTANCES, DISTORTIONS, '-o', color = '#d90000')
    plt.xlabel("Scaled distance")
    plt.ylabel("Scaled time distortion")

    plt.annotate('P05', xy=(DISTANCES[0], DISTORTIONS[0]), xytext=(4, 4), textcoords='offset pixels')

    for x, y, label in zip(DISTANCES[1:8], DISTORTIONS[1:8], POSSIBLE_STEP_PATTERNS[1:8]):
        plt.annotate(label[label.index('P'):], xy=(x, y), xytext=(5, 5), textcoords='offset pixels')

    for x, y, label in zip(DISTANCES[10::3], DISTORTIONS[10::3], POSSIBLE_STEP_PATTERNS[10::3]):
        plt.annotate(label[label.index('P'):], xy=(x, y), xytext=(0, 5), textcoords='offset pixels')
    plt.tight_layout()
    plt.show()
