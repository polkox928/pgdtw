# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 09:35:10 2019

@author: DEPAGRA
"""

import libdtw as lib
import pickle
import numpy as np
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import concordance_index_censored
from sksurv.ensemble import GradientBoostingSurvivalAnalysis
import matplotlib.pyplot as plt
import time
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

data = lib.load_data()

D = lib.Dtw(data)

step_pattern = 'symmetricP2'
query_id = '5379'

data_set = D.generate_train_set(500, step_pattern, n_jobs = -1)
data_set.drop(columns = ['ref_len'], inplace = True)
for idx, row in data_set.iterrows():
    query_id = row['query_id']
    batch = D.data['queries'][query_id]
    length = row['length']
    for pv in batch:
        data_set.at[idx, pv['name']] = pv['values'][length - 1]

data_set = pd.read_csv('data_set2500.csv', header = 0)
data_set = data_set.loc[:, online_surv.columns]
data_set.drop(columns = 'ba_PLAu1m2', inplace = True)
data_set.drop(columns = 'ba_TCfg3Yxn', inplace = True)
for idx, row in data_set.iterrows():
    query_id = row['query_id']
    batch = D.data['queries'][query_id]
    length = row['length']
    for pv in batch:
        data_set.at[idx, pv['name']] = pv['values'][length - 1]
        
online = D.generate_train_set(step_pattern=step_pattern, query_id=query_id, n_jobs = -1)

def build_structured_array(data_set):
    output = list()
    for idx, row in data_set.iterrows():
        survival_time = row['true_length'] - row['length']
        output.append((True, survival_time))
    res = np.array(output, dtype = [('status', bool), ('time_remaining', 'f8')])
    return res

survival = build_structured_array(data_set)
data_set.drop(columns=['true_length', 'query_id', 'step_pattern'], inplace = True)
data_y = build_structured_array(data_set)[400:]
data_set_surv = data_set.loc[:399].drop(columns=['true_length', 'query_id', 'step_pattern', 'ref_len'])
online_surv = online.drop(columns=['true_length', 'query_id', 'step_pattern', 'ref_len'])
test_surv = data_set.loc[400:].drop(columns=['true_length', 'query_id', 'step_pattern', 'ref_len'])

estimator = CoxPHSurvivalAnalysis()

estimator.fit(data_set_surv, survival)

prediction = estimator.predict(test_surv)

result = concordance_index_censored(data_y["status"], data_y["time_remaining"], prediction)
result[0]

pred_surv = estimator.predict_survival_function(test_surv.loc[400,])
pred_surv[0].y = np.gradient(1 - pred_surv[0].y)


for i, c in enumerate(pred_surv):
    plt.plot(c.x, c.y)
plt.ylabel("est. probability of survival $\hat{S}(t)$")
plt.xlabel("time $t$")
data_y[0]

#%%
# DROP CORRELATED FEATURES
# Create correlation matrix
corr_matrix = data_set.corr().abs()

# Select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

# Find index of feature columns with correlation greater than 0.95
to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]

# Drop features 
data_set_clean = data_set.drop(columns = to_drop) 
#%%
step_pattern = 'symmetricP2'
query_id = '5426'
online = D.generate_train_set(step_pattern=step_pattern, query_id=query_id, n_jobs = -1)
for idx, row in online.iterrows():
    _id = row['query_id']
    batch = D.data['queries'][_id]
    length = row['length']
    for pv in batch:
        online.at[idx, pv['name']] = pv['values'][length - 1]

true = len(data[query_id][0]['values'])
#%%
online_surv = online.loc[:, data_set_clean.columns]#online.drop(columns=['true_length', 'query_id', 'step_pattern', 'ref_len'])
online_surv.drop(columns = 'ba_PLAu1m2', inplace = True)

#%%
estimator = CoxPHSurvivalAnalysis(n_iter=500)
estimator.fit(data_set_clean, survival)
#%%
i = 0
fig = plt.figure()
plt.ion()
#%%
plt.cla()
row = online_surv.loc[i]    
pred_surv = estimator.predict_survival_function(row)
x = np.array(pred_surv[0].x) + i+1 
y = 1-pred_surv[0].y
plt.plot(x, y)

plt.ylabel("est. probability of survival $\hat{S}(t)$")
plt.xlabel("time $t$")
plt.vlines(true, 0, 1)
plt.xlim((0, 500))
plt.ylim((0, 1))
i += 10



#%%
scaler = MinMaxScaler()
scaler.fit(data_set_clean)

data_set_scale = pd.DataFrame(scaler.transform(data_set_clean), columns = data_set_clean.columns)
online_surv_scale = pd.DataFrame(scaler.transform(online_surv), columns = data_set_clean.columns)
