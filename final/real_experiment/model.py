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
    """
    Starting from a usual dataset, this function creates a structured numpy array of 2-tuples, where
    the first entry represent the 'status' of the entry (censored or no event = False, event = True)
    and the second represents the time to event
    Parameters
    ----------
    data_set: Pandas data frame
                Data set containing at least the length of the online query and its total duration

    Returns
    -------
    res : Numpy structured array
                Array suitable to be used by sksurv methods
    """
    output = list()
    for idx, row in data_set.iterrows():
        survival_time = row['true_length'] - row['length']
        output.append((True, survival_time))
    res = np.array(output, dtype=[('status', bool), ('time_remaining', 'f8')])
    return res


def generate_dataset_xy(t_ref, t, ongoing_id, D, data):
    """
    Generates a data set relative to the ongoing batch. It takes the time 't' on the ongoing batch,
    the mapped 't_ref' on the reference batch and than every point in the historical batches that
    were mapped to 't_ref'. It adds also the PV values to every entry

    Parameters
    ----------
    t_ref : int
                Time instant on the reference batch
    t : int
                Time instant on the query batch
    ongoing_id : string
                ID of the ongoing query
    D : Dtw object
                Dtw object with open-ended information
    data : dict
                Dictionary of the form {batch_ID : list_ov_PVs_dictionaries}

    Returns
    -------
    tuple
            (data_set, data_y)
            data_set : Pandas data frame
                        Data set containing information about:
                        - DTW distance
                        - length
                        - PV values
            data_y : Numpy structured array
                        Structured array suitable to be used with sksurv methods
    """
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
    """
    Methods
    -------
    - __init__
    - fit
    - predict
    - score
    - get_params
    - set_params
    """

    def __init__(self, dtw_obj, regressor=LinearRegression(), loss='coxph', learning_rate=0.1,\
                                    n_estimators=100, max_depth=3, subsample=1.0, random_state=42):
        """
        The class needs to be initialized with the Dtw object already trained, the regression model
        to use after the survival analysis model, and all the parameters for the survival analysis
        model

        Parameters
        ----------
        dtw_obj : Dtw object
                    Trained Dtw object
        regressor : sklearn model
                    Sklearn regression model
        loss, learning_rate, n_estimators, max_depth, subsample :
                    parameters of the GradientBoostingSurvivalAnalysis method
                    Complete DOC : https://scikit-survival.readthedocs.io/en/latest/generated/sksurv.ensemble.GradientBoostingSurvivalAnalysis.html
        random_state : int
                    Seed of the pseudo random number generator
        """
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
        """
        Fits the survival analysis model to the training data, and prepare the risk data used in the
        predict method by the regression model

        Parameters
        ----------
        x_train : pandas data frame
                    data set of predictors as returned by generate_dataset_xy()
        y_train : numpy structured array
                    Structured array suitable to be used with sksurv methods

        Returns
        -------
        Reference to the object itself
        """
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
        """
        First computes the risk of the new data point, than converts it to a time measure via the
        regression model feeded as input to the class

        Parameters
        ----------
        new_x : pandas data frame
                    Data frame of the data point to predict
        by : string {'rank', 'risk', 'scaled_risk'}
                    Which feature to consider when applying the regression model to predict
                                                                                    the time-to-end

        Returns
        -------
        numpy array
                    Array of time-to-end estimates

        """
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
        """
        Computes the mean absolute erros on the test set in input

        Parameters
        ----------
        x_test : pandas data frame
                    data set of predictors as returned by generate_dataset_xy()
        y_test : numpy structured array
                    Structured array suitable to be used with sksurv methods

        Returns
        -------
        float
                    mean absolute error
        """
        y_pred = self.predict(x_test)
        return np.mean(np.abs(y_pred - y_test['time_remaining']))

    def get_params(self, deep=True):
        """
        Returns the parameters of the class

        Parameters:
        deep : boolean
                    Inserted only for compatibility with sklearn

        Returns
        dict
                    Dictionary of the initializing parameters of the Estimator class
        """
        return {'dtw_obj': self.dtw_obj,
                'regressor': self.regressor,
                'loss': self.loss,
                'learning_rate': self.learning_rate,
                'n_estimators': self.n_estimators,
                'max_depth': self.max_depth,
                'subsample': self.subsample}

    def set_params(self, parameters):
        """
        Sets the parameters of the class
        
        Parameters
        ----------
        parameters : dict
                    Dictionary of pairs {parameter_name : parameter_value}
        """
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
