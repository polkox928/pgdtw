# -*- coding: utf-8 -*-
"""
Created on Fri Feb  1 14:27:34 2019

@author: DEPAGRA
"""

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
step_pattern = 'symmetricP2'

D = lib.Dtw(data)

with open('dtwObjOptWeights16AllFeats.pickle', 'rb') as f:
    D_weights = pickle.load(f)
D.data['feat_weights'] = D_weights

#try:
#    with open('data/all_align100.pickle', 'rb') as f:
#        D.data['warp_dist'] = pickle.load(f)
#except OSError as ex:
#    for _id in tqdm(D.data['queriesID']):
#        D.call_dtw(_id, step_pattern=step_pattern, n_jobs=1)
#
#    with open('data/all_align100.pickle', 'wb') as f:
#        pickle.dump(D.data['warp_dist'], f, protocol=pickle.HIGHEST_PROTOCOL)

try:
    with open('data/all_sub100.pickle', 'rb') as f:
        D.data_open_ended['queries'] = pickle.load(f)
except OSError as ex:
    for _id in tqdm(D.data['queriesID']):
        D.call_dtw(_id, step_pattern=step_pattern, n_jobs=1, open_ended=True, all_sub_seq=True)

    with open('data/all_sub100.pickle', 'wb') as f:
        pickle.dump(D.data_open_end['queries'], f, protocol=pickle.HIGHEST_PROTOCOL)





true_length = len(data[online_id][0]['values'])
for online_id in D.data['queriesID']:
    try:
        online_raw = pd.read_csv('online_data_sets/online_%s.csv'%online_id, header=0, index_col=None, dtype={'query_id': str})
    except OSError as ex:
        print(online_id)
        online_raw = D.generate_train_set(step_pattern=step_pattern, query_id=online_id, n_jobs = -1)
        online_raw.to_csv('online_data_sets/online_%s.csv'%online_id, header=True, index = False,)
#%%
online_id = '5243'
try:
    online_raw = pd.read_csv('online_data_sets/online_%s.csv'%online_id, header=0, index_col=None, dtype={'query_id': str})
except OSError as ex:
    print(online_id)
    online_raw = D.generate_train_set(step_pattern=step_pattern, query_id=online_id, n_jobs = -1)
    online_raw.to_csv('online_data_sets/online_%s.csv'%online_id, header=True, index = False,)
estimates = list()
for online_t in online_raw['length'].values[50:]:
    print(online_t)
    t_prime = online_raw.loc[online_raw['length']==online_t, 'ref_prefix'].values[0] - 1

    try:
        online_raw.index = online_raw['query_id']
        online = online_raw.drop(columns=['query_id', 'ref_len', 'step_pattern'])

    except: pass


    data_set = list()

    for _id, warp_dist in D.data['warp_dist'].items():
        if _id != online_id:
            mapped_points = list(filter(lambda x: x[0]==t_prime, warp_dist))
            for (i, j, d) in mapped_points:
                data_point = {'DTW_distance': d,
                              'length': j + 1,
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
    data_set_new['time_remaining'] = data_y_new['time_remaining']

    data_set_new.sort_values(by='risk', ascending=False, inplace=True)

    t_left = data_set_new['time_remaining']
    risks = data_set_new['risk']
    loc = data_set_new.index.get_loc(online_id)+1
#        plt.figure()
#        plt.bar(np.arange(1, len(t_left)+1), t_left.values)
#        plt.bar(loc, t_left[online_id], color = 'r')
#        plt.xticks(np.arange(1, len(t_left)+1), data_set_new.index, rotation = 90)
#        plt.show()
    xy = [(x,y) for x, y in zip(np.arange(1, len(t_left)), t_left.values) if x != loc]
    x = np.array([x[0] for x in xy]).reshape(-1,1)
    y = np.array([x[1] for x in xy]).reshape(-1,1)
    lm = LinearRegression().fit(x, y)
    est = lm.predict(loc)
    estimates.append(est[0][0]+online_t)

plt.figure()
plt.plot(np.arange(51, true_length+1), estimates)
plt.plot(x=[51, true_length], y=[51, true_length], color = "gray")
plt.hlines(true_length, 51, 51+len(estimates))
plt.xlim((51, true_length))
plt.title(online_id)
plt.show()
