# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 09:31:36 2019

@author: DEPAGRA
"""

import libdtw as lib
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.neural_network import MLPRegressor
from tqdm import tqdm

data = lib.load_data(50)
data['reference'] = '5149'
D = lib.Dtw(data) 

#file_path = 'dtwObjOptWeights21AllFeats.pickle'
#
#with open(file_path, 'rb') as f:
#    D.data['feat_weights'] = pickle.load(f)
step_pattern = 'symmetricP2'
data_set = D.generate_train_set(500, step_pattern, n_jobs = -1)
#data_set.to_csv('data_set2500.csv', index = False)
#data_set = data_set.loc[:, ['DTW_distance', 'length', 'query_id', 'ref_len', 'ref_prefix', 'step_pattern', 'true_length']]
for idx, row in data_set.iterrows():
    query_id = row['query_id']
    batch = D.data['queries'][query_id]
    length = row['length']
    for pv in batch:
        data_set.at[idx, pv['name']] = pv['values'][length - 1]
        
#X = data_set.loc[:, [col for col in data_set.columns if (col != 'true_length' and col != 'step_pattern' and col != 'query_id')]].values
#y = data_set.loc[:, ['true_length']].values
#
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)
#
#rf = RandomForestRegressor()
#
#cross_validate(rf,  X, y, verbose = 5, scoring = make_scorer(mean_squared_error))        
#
#rf.fit(X_train, y_train.ravel())
#
#rf.predict(X_test)
#
#np.sqrt(np.mean((y_test.ravel() - rf.predict(X_test))**2))
#
#data_set.loc[:, ['length']].max()

def make_train_test(data_set, test_query_id, verbose = False):
    train = data_set.loc[data_set['query_id'] != test_query_id, :].drop(columns= ['step_pattern', 'query_id'])
    test = data_set.loc[data_set['query_id'] == test_query_id, :].drop(columns= ['step_pattern', 'query_id'])
    
    X_train = train.drop(columns = ['true_length'])
    y_train = train.loc[:, ['true_length']]
    
    X_test = test.drop(columns = ['true_length'])
    y_test = test.loc[:, ['true_length']]
    
    if verbose: print('Train samples: %d\nTest samples: %d'%(train.shape[0], test.shape[0]))
    
    return X_train, X_test, y_train, y_test

#X_train, X_test, y_train, y_test = make_train_test(data_set, '5300')
#
#rf = RandomForestRegressor()
#
#rf.fit(X_train, y_train.values.ravel())
#
#rf.predict(X_test)
#
#np.sqrt(np.mean((y_test.values.ravel() - rf.predict(X_test))**2))
#    
#    
#feat_importances = pd.Series(rf.feature_importances_, index=X_train.columns)
#feat_importances.nlargest(50).plot(kind='barh')
#plt.title('Feature Importances')
#plt.xlabel('Relative Importance')
#plt.tight_layout()
#plt.show()
#
#fig = plt.figure()
#D.plot_by_name('5300', 'ba_PCPUSq5ah')

scores = dict()
rf = RandomForestRegressor(random_state=42)
for _id in tqdm(D.data['queriesID']):
    X_train, X_test, y_train, y_test = make_train_test(data_set, _id)

    
    rf.fit(X_train.values, y_train.values.ravel())
    
    rf.predict(X_test)
    
    scores[_id] = np.sqrt(np.mean((y_test.values.ravel() - rf.predict(X_test))**2))
    
    
ids = [key for key in data.keys() if key != 'reference' and key != data['reference']]
true_length = [(_id, len(data[_id][0]['values'])) for _id in ids]
sorted(true_length, key = lambda x: x[1])
errs = [scores[_id] for _id in ids]

plt.scatter([x[1] for x in true_length], errs)

#%%
query_id = '5379'
online = D.generate_train_set(step_pattern=step_pattern, query_id=query_id, n_jobs = -1)
for idx, row in online.iterrows():
    query_id = row['query_id']
    batch = D.data['queries'][query_id]
    length = row['length']
    for pv in batch:
        online.at[idx, pv['name']] = pv['values'][length - 1]
        
X_train, X_test, y_train, y_test = make_train_test(data_set, query_id)
X_test = online.drop(columns = ['true_length', 'step_pattern', 'query_id'])
y_test = online.loc[:, ['true_length']]
rf.fit(X_train.values, y_train.values.ravel())
y_pred = rf.predict(X_test)

plt.plot(np.arange(1, len(y_pred)+1), y_pred)
plt.hlines(y = len(y_pred), xmin = 1, xmax = len(y_pred))
plt.show()

#%%

step_pattern = 'symmetricP2'


data1 = lib.load_data(50)
data1['reference'] = '5153'
A = lib.Dtw(data1)

data_set1 = A.generate_train_set(500, step_pattern, n_jobs = -1)
for idx, row in data_set1.iterrows():
    query_id = row['query_id']
    batch = A.data['queries'][query_id]
    length = row['length']
    for pv in batch:
        data_set1.at[idx, pv['name']] = pv['values'][length - 1]



data2 = lib.load_data(50)
data2['reference'] = '5415'
B = lib.Dtw(data2) 

data_set2 = B.generate_train_set(500, step_pattern, n_jobs = -1)
for idx, row in data_set2.iterrows():
    query_id = row['query_id']
    batch = B.data['queries'][query_id]
    length = row['length']
    for pv in batch:
        data_set2.at[idx, pv['name']] = pv['values'][length - 1]

data3 = lib.load_data(50)
data3['reference'] = '5472'
C = lib.Dtw(data3) 


data_set3 = C.generate_train_set(500, step_pattern, n_jobs = -1)
for idx, row in data_set3.iterrows():
    query_id = row['query_id']
    batch = C.data['queries'][query_id]
    length = row['length']
    for pv in batch:
        data_set3.at[idx, pv['name']] = pv['values'][length - 1]
        
#%%
data_setAll = pd.concat([data_set1, data_set2, data_set3], ignore_index = True)

query_id = '5379'
onlineA = A.generate_train_set(step_pattern=step_pattern, query_id=query_id, n_jobs = -1)
for idx, row in onlineA.iterrows():
    query_id = row['query_id']
    batch = D.data['queries'][query_id]
    length = row['length']
    for pv in batch:
        onlineA.at[idx, pv['name']] = pv['values'][length - 1]

onlineB = B.generate_train_set(step_pattern=step_pattern, query_id=query_id, n_jobs = -1)
for idx, row in onlineB.iterrows():
    query_id = row['query_id']
    batch = B.data['queries'][query_id]
    length = row['length']
    for pv in batch:
        onlineB.at[idx, pv['name']] = pv['values'][length - 1]
        
onlineC = C.generate_train_set(step_pattern=step_pattern, query_id=query_id, n_jobs = -1)
for idx, row in onlineC.iterrows():
    query_id = row['query_id']
    batch = C.data['queries'][query_id]
    length = row['length']
    for pv in batch:
        onlineC.at[idx, pv['name']] = pv['values'][length - 1]
        
X_train, X_test, y_train, y_test = make_train_test(data_setAll, query_id)

X_test = onlineA.drop(columns = ['true_length', 'step_pattern', 'query_id'])
y_test = onlineA.loc[:, ['true_length']]
rf.fit(X_train.values, y_train.values.ravel())
y_pred = rf.predict(X_test)

plt.plot(np.arange(1, len(y_pred)+1), y_pred)
plt.hlines(y = len(y_pred), xmin = 1, xmax = len(y_pred))
plt.show()

X_test = onlineB.drop(columns = ['true_length', 'step_pattern', 'query_id'])
y_test = onlineB.loc[:, ['true_length']]
rf.fit(X_train.values, y_train.values.ravel())
y_pred = rf.predict(X_test)

plt.plot(np.arange(1, len(y_pred)+1), y_pred)
plt.hlines(y = len(y_pred), xmin = 1, xmax = len(y_pred))
plt.show()

X_test = onlineC.drop(columns = ['true_length', 'step_pattern', 'query_id'])
y_test = onlineC.loc[:, ['true_length']]
rf.fit(X_train.values, y_train.values.ravel())
y_pred = rf.predict(X_test)

plt.plot(np.arange(1, len(y_pred)+1), y_pred)
plt.hlines(y = len(y_pred), xmin = 1, xmax = len(y_pred))
plt.show()