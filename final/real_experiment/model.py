import libdtw as lib
from copy import copy, deepcopy
from tqdm import tqdm_notebook, tqdm
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from sksurv.ensemble import GradientBoostingSurvivalAnalysis
from sklearn.preprocessing import MinMaxScaler
from joblib import Parallel, delayed
import os


def build_structured_array(data_set):
    output = list()
    for idx, row in data_set.iterrows():
        survival_time = row['true_length'] - row['length']
        output.append((True, survival_time))
    res = np.array(output, dtype=[('status', bool), ('time_remaining', 'f8')])
    return res


def generate_dataset_xy(t_ref, t, ongoing_id, D, data):
    data_set = list()

    for _id, warp_dist in D.data_open_ended['warp_dist'].items():
        if _id == ongoing_id:
            mapped_points = list(filter(lambda x: (x[0] == t_ref and x[1] == t), warp_dist))
        else:
            mapped_points = list(filter(lambda x: x[0] == t_ref, warp_dist))
        for (i, j, d) in mapped_points:
            data_point = {'DTW_distance': d,
                          'length': j + 1,
                          'query_id': _id,
                          'true_length': len(data[_id][0]['values'])}
            data_set.append(data_point)

    data_set = pd.DataFrame(data_set)
    data_set.index = data_set['query_id']

    data_y = build_structured_array(data_set)
    data_set.drop(columns=['query_id', 'true_length'], inplace=True)

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
                                                      learning_rate=self.learning_rate,
                                                      n_estimators=self.n_estimators,
                                                      max_depth=self.max_depth,
                                                      subsample=self.subsample,
                                                      random_state=self.random_state)

        self.x_train = x_train
        self.y_train = y_train

        self.model.fit(self.x_train, self.y_train)

        self.data_set = pd.concat([self.x_train, pd.Series(
            data=self.y_train['time_remaining'], index=self.x_train.index, name='time_remaining')], axis=1, sort=False)
        self.data_set['risk'] = self.model.predict(self.x_train)

        return self

    def predict(self, new_x, by='risk'):
        x_new = pd.DataFrame(deepcopy(new_x))
        x_new['risk'] = self.model.predict(x_new)
        query_id = list(x_new.index)[0]
        x_length = len(self.dtw_obj.data['queries'][query_id][0]['values'])
        x_new['time_remaining'] = x_length - x_new['length']

        self.data_set_extd = pd.concat([self.data_set, x_new], axis=0, sort=False)
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
            xy = [(x, y) for (x, y) in zip(x_values.values, y_values.values) if x != loc]
            x = np.array([x[0] for x in xy]).reshape(-1, 1)
            y = np.array([x[1] for x in xy])
        # Add possibility for risk as X variable
            reg = self.regressor.fit(X=x, y=np.log1p(y))
            if by == 'scaled_risk':
                ests.append(np.expm1(reg.predict(scaler.transform(x_values.values[loc]))[0]))
            else:
                ests.append(np.expm1(reg.predict(x_values.values[loc])[0]))

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
