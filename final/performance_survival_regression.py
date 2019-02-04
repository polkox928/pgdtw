#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 17:33:16 2019

@author: paolograniero
"""

import libdtw as lib
import numpy as np
import pandas as pd
import pickle
import os
import matplotlib.pyplot as plt

data = lib.load_data(100)

estimates = dict()
true_length = dict()

folder_path = 'estimates_all_sub/'
for fn in os.listdir(folder_path):
    _id = fn.split('.')[0]
    true_length[_id] = len(data[_id][0]['values'])
    with open(folder_path+fn, 'rb') as f:
        estimates[_id] = pickle.load(f)

avg_length = np.mean(list(true_length.values()))

naive_performance = dict()
model_performance = dict()
moving_average_performance = dict()
absolute_better_than_naive = list()
average_better_than_naive = list()
interpolated_average_estimates = dict()
deviation_on_interpolation = dict()
dev_interp_list = list()
def mae(predictions, targets):
    return np.mean(np.abs(predictions-targets))

for _id, ests in estimates.items():
    true_len = true_length[_id]

    naive = mae(avg_length,true_len)
    model = mae(np.array(ests), true_len)
    average = mae(pd.Series(ests).rolling(30).mean().fillna(method='bfill').values,true_len)

    naive_performance[_id] = naive
    model_performance[_id] = model
    moving_average_performance[_id] = average

    absolute_better_than_naive.append(1*(model < naive))
    average_better_than_naive.append(1*(average < naive))

    interpolation = np.interp(np.arange(0,100), np.arange(0, len(ests)), ests)

    interpolated_average_estimates[_id] = interpolation
    deviation_on_interpolation[_id] = np.abs(interpolation - true_len)
    dev_interp_list.append(np.abs(interpolation - true_len))

dev_interp = np.mean(np.array(dev_interp_list), axis = 0)
avg_naive = np.mean(list(naive_performance.values()))
plt.boxplot(np.array(dev_interp_list))
plt.hlines(avg_naive, 0, 100)
plt.xticks(rotation = 90)
plt.show()
print('Fraction of absolute-better-than-naive: %0.3f'%np.mean(np.array(absolute_better_than_naive)))
print('Fraction of average-better-than-naive: %0.3f'%np.mean(np.array(average_better_than_naive)))


# Interpolation























