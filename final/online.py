# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 11:30:56 2019

@author: DEPAGRA
"""

#from grid_search_surv_regr import Estimator
import libdtw as lib
from copy import copy, deepcopy
from tqdm import tqdm_notebook, tqdm
import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd
from sksurv.ensemble import GradientBoostingSurvivalAnalysis
from sklearn.preprocessing import MinMaxScaler

def build_structured_array(data_set):
    output = list()
    for idx, row in data_set.iterrows():
        survival_time = row['true_length'] - row['length']
        output.append((True, survival_time))
    res = np.array(output, dtype = [('status', bool), ('time_remaining', 'f8')])
    return res

def generate_dataset_xy(t_ref, D):
    data_set = list()

    for _id, warp_dist in D.data_open_ended['warp_dist'].items():
        mapped_points = list(filter(lambda x: x[0]==t_ref, warp_dist))
        for (i, j, d) in mapped_points:
            data_point = {'DTW_distance': d,
                          'length': j + 1,
                          'query_id' : _id,
                          'true_length': len(data[_id][0]['values'])}
            data_set.append(data_point)
        
    data_set = pd.DataFrame(data_set)
    data_set.index = data_set['query_id']
    
    data_y = build_structured_array(data_set)
    data_set.drop(columns = ['query_id', 'true_length'], inplace = True)

    for _id, row in data_set.iterrows():
        batch = D.data['queries'][_id]
        length = int(row['length'])
        for pv in batch:
            data_set.at[_id, pv['name']] = pv['values'][length - 1]
    
    return (data_set, data_y)

class Estimator:
    
    def __init__(self, dtw_obj, regressor=LinearRegression(), loss='coxph', learning_rate=0.1, n_estimators=100, max_depth=3, subsample=1.0, random_state=42):
        np.random.seed(random_state)
        self.regressor = regressor
        self.loss = loss
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.subsample = subsample
        self.random_state = random_state
        
        
        self.dtw_obj = dtw_obj
        
    def fit(self, x_train, y_train):
        self.model = GradientBoostingSurvivalAnalysis(loss=self.loss,
                                                 learning_rate = self.learning_rate,
                                                 n_estimators=self.n_estimators,
                                                 max_depth=self.max_depth,
                                                 subsample=self.subsample,
                                                 random_state = self.random_state)
        
        self.x_train = x_train
        self.y_train = y_train
        
        self.model.fit(self.x_train, self.y_train)
        
        self.data_set = pd.concat([self.x_train, pd.Series(data=self.y_train['time_remaining'], index=self.x_train.index, name='time_remaining')], axis=1, sort = False)
        self.data_set['risk'] = self.model.predict(self.x_train)
        
        return self
    
    def predict(self, new_x, by='risk'):
        x_new = pd.DataFrame(deepcopy(new_x))
        x_new['risk'] = self.model.predict(x_new)
        query_id = list(x_new.index)[0]
        x_length = len(self.dtw_obj.data['queries'][query_id][0]['values'])
        x_new['time_remaining'] = x_length -x_new['length']
        
        self.data_set_extd = pd.concat([self.data_set, x_new], axis = 0, sort = False)
        self.data_set_extd.sort_values(by='risk', ascending=False, inplace=True)

        locations = self.data_set_extd.index.get_loc(query_id)
        
        locs = list()
        if type(locations) == slice:
            start, stop = locations.start, locations.stop
            locs.extend([loc for loc in np.arange(start, stop)])
        elif type(locations) == int or type(locations) == np.int64:
            locs = [locations]
        elif type(locations) == np.ndarray:
            locs = np.arange(len(locations))[locations]
        else:
            print('ERROR')
            print(type(locations))
            locs = []
          
        y_values = self.data_set_extd['time_remaining']
        
        if by == 'rank':
            x_values = pd.Series(np.arange(y_values))
        elif by == 'risk':
            x_values = self.data_set_extd['risk']
        elif by == 'scaled_risk':
            scaler = MinMaxScaler()
            x_values = scaler.fit_transform(self.data_set_extd['risk'])
        ests = list()
        
        for loc in locs:
#            print(locs)
#            print([x for x, y in zip(np.arange(len(t_left)), t_left.values)])
            xy = [(x,y) for (x, y) in zip(x_values.values, y_values.values) if x != loc]
            x = np.array([x[0] for x in xy]).reshape(-1,1)
            y = np.array([x[1] for x in xy])
        ## Add possibility for risk as X variable
            reg = self.regressor.fit(X=x, y=y)
            if by == 'scaled_risk':
                ests.append(reg.predict(scaler.transform(x_values.values[loc]))[0])
            else:
                ests.append(reg.predict(x_values.values[loc])[0])
            
        return np.array(ests)
    
    def score(self, x_test, y_test):
        y_pred = self.predict(x_test)
        return np.mean(np.abs(y_pred - y_test['time_remaining']))
    
    def get_params(self, deep=True):
        return {'dtw_obj': self.dtw_obj,
                'regressor': self.regressor,
                'loss': self.loss,
                'learning_rate': self.learning_rate,
                'n_estimators': self.n_estimators,
                'max_depth': self.max_depth,
                'subsample': self.subsample}

    def set_params(self, parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
            
all_data = lib.load_data(1000)
_ = all_data.pop('reference')

ids = list(all_data.keys())

data = dict()
for _id in ids[:10]:
    data[_id] = copy(all_data[_id])
    
data = lib.assign_ref(data)

step_pattern = 'symmetric2'
p_max = list()
for idx in range(11, len(ids)):
    ongoing_id = ids[idx]
    hist_data = {_id : copy(all_data[_id]) for _id in ids[:idx-1]}
    data = lib.assign_ref(hist_data)
    data[ongoing_id] = copy(all_data[ongoing_id])
    D = lib.Dtw(data)
    if len(D.data['queries']) < 5:
        continue
    p_max.append(D.get_global_p_max())
    for _id in tqdm(D.data['queriesID']):
        D.call_dtw(_id, step_pattern=step_pattern, n_jobs=1, open_ended=True, all_sub_seq=True)
        
    estimator = Estimator(dtw_obj=D)
    for (i, j, d) in tqdm(D.data_open_ended['warp_dist'][ongoing_id], desc='%d/%d'%(idx, len(ids)+1)):
        data_x, data_y = generate_dataset_xy(i, D)
        
        train_id = data_x.loc[data_x.index != ongoing_id, :].index
        test_id = data_x.loc[data_x.index == ongoing_id, :].index
        
        train_loc = list()
        for _id in train_id.unique():
            locs = data_x.index.get_loc(_id)
            if type(locs) == slice:
                start, stop = locs.start, locs.stop
                train_loc.extend([[loc] for loc in np.arange(start, stop)])
            elif type(locs) == int or type(locs) == np.int64:
                train_loc.append([locs])
            else: print('\n', locs, type(locs))
       
        if type(data_x.index.get_loc(ongoing_id)) == slice:
            locs = data_x.index.get_loc(ongoing_id)
            start, stop = locs.start, locs.stop
            test_loc = [[loc] for loc in np.arange(start, stop)]
        elif type(data_x.index.get_loc(ongoing_id)) == int or type(data_x.index.get_loc(ongoing_id)) == np.int64:
            test_loc = [data_x.index.get_loc(ongoing_id)]
        else: print('ERROR 2')
        
        train_id, test_id, train_loc, test_loc = train_id.unique(), test_id.unique(), train_loc, test_loc
        if len(train_id) < 2:
            continue
        x_train = data_x.loc[train_id, :]

        y_train = np.array(np.concatenate([data_y[idx] for idx in train_loc], axis=0), dtype = [('status', bool), ('time_remaining', 'f8')])

        x_test = data_x.loc[test_id, :]
        y_test_raw = [data_y[idx] for idx in test_loc]

        y_test = np.concatenate(y_test_raw, axis=0) if len(y_test_raw)>1 else y_test_raw
        y_test = np.array(y_test, dtype = [('status', bool), ('time_remaining', 'f8')])
        
        estimator.fit(x_train, y_train)

        with open('cv_online/%s.csv'%test_id.values[0], 'a+') as f:
            y_pred = estimator.predict(x_test)
            for (_id, row), y_p in zip(x_test.iterrows(), y_pred):
                f.write('%s, %d, %d, %0.1f\n'%(test_id.values[0], row['length'], len(D.data['queries'][test_id.values[0]][0]['values']), y_p))