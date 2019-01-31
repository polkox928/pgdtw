# -*- coding: utf-8 -*-
import libdtw as lib
import pickle
import numpy as np
from sksurv.linear_model import CoxPHSurvivalAnalysis
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from seaborn import kdeplot
import pymc3

data = lib.load_data(100)

D = lib.Dtw(data)

with open('dtwObjOptWeights16AllFeats.pickle', 'rb') as f:
    D_weights = pickle.load(f)
D.data['feat_weights'] = D_weights

step_pattern = 'symmetricP2'

data_set = D.generate_train_set(500, step_pattern, n_jobs = -1)

query_id = '5300'
true_length = len(data[query_id][0]['values'])
online5472 = D.generate_train_set(step_pattern=step_pattern, query_id=query_id, n_jobs = -1)

def build_structured_array(data_set):
    output = list()
    for idx, row in data_set.iterrows():
        survival_time = row['true_length'] - row['length']
        output.append((True, survival_time))
    res = np.array(output, dtype = [('status', bool), ('time_remaining', 'f8')])
    return res

data_y = build_structured_array(data_set)
data_set_surv = data_set.drop(columns=['true_length', 'query_id', 'step_pattern', 'ref_len'])
online_surv = online.loc[:, data_set_surv.columns]
estimator = CoxPHSurvivalAnalysis()

estimator.fit(data_set_surv, data_y)

#%%
i = 0
fig = plt.figure()
plt.ion()
#%%
plt.cla()
row = online_surv.loc[i]    
pred_surv = estimator.predict_survival_function(row)
x = np.array(pred_surv[0].x) + i+1 
y_weight = np.gradient(1 - pred_surv[0].y, x)
scaler = MinMaxScaler((0, 100))
y = np.floor(scaler.fit_transform(y_weight.reshape(-1, 1)).ravel())
x_data = list()
for j, n in zip(x, y):
    for k in np.arange(n):
        x_data.append(j)
kdeplot(x_data)
low, high = np.quantile(x_data, [.025, .975])

plt.ylabel("est. probability of survival $\hat{S}(t)$")
plt.xlabel("time $t$")
plt.vlines(true_length, 0, 1)
plt.vlines(low, 0, 1, color = "b")
plt.vlines(high, 0, 1, color = "b")
plt.vlines(i, 0, 1, color = "r")
plt.xlim((0, 500))
#plt.ylim((0, 1))
i += 5