# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 11:28:58 2019

@author: DEPAGRA
"""
import libdtw as lib
from tqdm import tqdm
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sksurv.ensemble import GradientBoostingSurvivalAnalysis as gbsa
from sklearn.linear_model import LinearRegression

data = lib.load_data(100)

D = lib.Dtw(data)

with open('dtwObjOptWeights16AllFeats.pickle', 'rb') as f:
    D_weights = pickle.load(f)
D.data['feat_weights'] = D_weights

step_pattern = 'symmetricP2'

if __name__ == '__main__':
    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"
    
    for _id in tqdm(D.data['queriesID']):
        D.call_dtw(_id, step_pattern=step_pattern, n_jobs=1)

#%%
online_id = '5300'

online = D.generate_train_set(step_pattern=step_pattern, query_id=online_id, n_jobs = -1)
#%%
estimates = list()
for online_t in range(50, len(online)):
    print(online_t)
    t_prime = online.loc[online['length']==online_t, 'ref_prefix'].values[0]
    
    try:
        online.index = online['query_id']
        online.drop(columns=['query_id'], inplace = True)
        online.drop(columns=['ref_len', 'step_pattern'], inplace=True)
    except: pass
    
    
    data_set = list()
    
    for _id, warp_dist in D.data['warp_dist'].items():
        if _id != online_id:
            mapped_points = list(filter(lambda x: x[0]==t_prime, warp_dist))
            for (i, j, d) in mapped_points:
                data_point = {'DTW_distance': d,
                              'length': j+1,
                              'query_id' : _id,
                              'true_length': len(data[_id][0]['values'])}
                data_set.append(data_point)
            
    def build_structured_array(data_set):
        output = list()
        for idx, row in data_set.iterrows():
            survival_time = row['true_length'] - row['length']
            output.append((True, survival_time))
        res = np.array(output, dtype = [('status', bool), ('time_remaining', 'f8')])
        return res
    
    data_set = pd.DataFrame(data_set)
    data_set.index = data_set['query_id']
    data_y = build_structured_array(data_set)
    data_set.drop(columns=['query_id'], inplace = True)
    
    for _id, row in data_set.iterrows():
        batch = D.data['queries'][_id]
        length = int(row['length'])
        for pv in batch:
            data_set.at[_id, pv['name']] = pv['values'][length - 1]
            
    corr_matrix = data_set.corr().abs()
    corr_matrix.fillna(1, inplace = True)
    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    
    # Find index of feature columns with correlation greater than 0.95
    to_drop = [column for column in upper.columns if any(upper[column] > 0.99) and column != 'true_length']
    
    # Drop features 
    data_set.drop(columns = to_drop, inplace = True)
    #print(data_set.shape, '\n', data_set.columns)
    
    data_x = data_set.drop(columns = ['true_length'])
    
    model = gbsa()
    model.fit(data_x.reindex(sorted(data_x.columns), axis=1), data_y)
    score = model.score(data_x.reindex(sorted(data_x.columns), axis=1), data_y); print(score)        
            
    data_set_risk = data_set
    data_set_risk['risk'] = model.predict(data_x.reindex(sorted(data_x.columns), axis=1))     
    data_set_risk.sort_values(by='risk', ascending=False, inplace=True)
    
    lengths = data_set_risk['true_length']
    #    fig = plt.figure()
    #    plt.bar(np.arange(1, len(lengths)+1), lengths.values)
    #    plt.show()
    
    
    for _id, row in online.iterrows():
        batch = D.data['queries'][_id]
        length = int(row['length'])
        for pv in batch:
            online.at[_id, pv['name']] = pv['values'][length - 1]
    
    online_clean = online.loc[:, data_x.columns]
    new_x = online_clean.loc[online['length'] == online_t, :]
    new_x['risk'] = model.predict(new_x.reindex(sorted(data_x.columns), axis=1))
    new_x['true_length'] = len(data[online_id][0]['values'])
    
    data_set_new = pd.concat([data_set, new_x])
    data_y_new = build_structured_array(data_set_new)
    data_x_new = data_set_new.drop(columns=['true_length', 'risk'])
    
    risk_complete = model.predict(data_x_new.reindex(sorted(data_x.columns), axis=1))
    new_score = model.score(data_x_new.reindex(sorted(data_x.columns), axis=1), data_y_new); print(new_score)
    data_set_new['risk'] = risk_complete
    
    data_set_new.sort_values(by='risk', ascending=False, inplace=True)
    
    lengths = data_set_new['true_length']
    risks = data_set_new['risk']
    loc = data_set_new.index.get_loc(online_id)+1
    #    plt.figure()
    #    plt.bar(np.arange(1, len(lengths)+1), lengths.values)
    #    plt.bar(loc, lengths[online_id], color = 'r')
    #    plt.xticks(np.arange(1, len(lengths)+1), data_set_new.index, rotation = 90)
    #    plt.show()
    #    
    xy = [(x,y) for x, y in zip(risks.values, lengths.values) if x != loc]
    x = np.array([x[0] for x in xy]).reshape(-1,1)
    y = np.array([x[1] for x in xy]).reshape(-1,1)
    lm = LinearRegression().fit(x, y)
    est = lm.predict(loc)
    estimates.append(est)
ests = [x[0][0] for x in estimates]
plt.plot(np.abs(np.array(ests)-415))
