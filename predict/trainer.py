#!/usr/bin/python
# -*- coding: utf-8 -*-

import json
import sys
import math
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression

data = pd.read_csv('/Users/chyc/Workspaces/Data/lottery/ssq.num', sep='\t',
                   names=['no', 'red1num', 'red2num', 'red3num', 'red4num', 'red5num', 'red6num', 'bluenum'])
data = data.fillna(0)
data = data.sort_values(by='no', ascending=True)
data['index'] = data.index
total = len(data)
extend_data = pd.DataFrame({'num': range(1, 34, 1)}, index=range(1, 34, 1))

extend_data['red1num_freq'] = data.groupby(by='red1num')['red1num'].count()
extend_data['red2num_freq'] = data.groupby(by='red2num')['red2num'].count()
extend_data['red3num_freq'] = data.groupby(by='red3num')['red3num'].count()
extend_data['red4num_freq'] = data.groupby(by='red4num')['red4num'].count()
extend_data['red5num_freq'] = data.groupby(by='red5num')['red5num'].count()
extend_data['red6num_freq'] = data.groupby(by='red6num')['red6num'].count()
extend_data['bluenum_freq'] = data.groupby(by='bluenum')['bluenum'].count()

extend_data['red1num_prob'] = extend_data['red1num_freq'] / total
extend_data['red2num_prob'] = extend_data['red2num_freq'] / total
extend_data['red3num_prob'] = extend_data['red3num_freq'] / total
extend_data['red4num_prob'] = extend_data['red4num_freq'] / total
extend_data['red5num_prob'] = extend_data['red5num_freq'] / total
extend_data['red6num_prob'] = extend_data['red6num_freq'] / total
extend_data['bluenum_prob'] = extend_data['bluenum_freq'] / total

extend_data = extend_data.fillna(0)

bluenum_features = pd.merge(left=data[['index', 'no', 'bluenum']], right=extend_data[['num', 'bluenum_freq', 'bluenum_prob']], left_on='bluenum',
                            right_on='num',
                            how='inner', sort=False)

bluenum_features['pre1index'] = bluenum_features['index'] - 1
bluenum_features['pre2index'] = bluenum_features['index'] - 2
bluenum_features['pre3index'] = bluenum_features['index'] - 3
bluenum_features['pre4index'] = bluenum_features['index'] - 4
bluenum_features['pre5index'] = bluenum_features['index'] - 5

bluenum_features = pd.merge(left=bluenum_features, right=data, left_on='pre1index', right_on='index', how='inner', suffixes=['', '-1'])
bluenum_features = pd.merge(left=bluenum_features, right=data, left_on='pre2index', right_on='index', how='inner', suffixes=['', '-2'])
bluenum_features = pd.merge(left=bluenum_features, right=data, left_on='pre3index', right_on='index', how='inner', suffixes=['', '-3'])
bluenum_features = pd.merge(left=bluenum_features, right=data, left_on='pre4index', right_on='index', how='inner', suffixes=['', '-4'])
bluenum_features = pd.merge(left=bluenum_features, right=data, left_on='pre5index', right_on='index', how='inner', suffixes=['', '-5'])

features = ['bluenum_freq', 'bluenum_prob',
            'red1num', 'red2num', 'red3num', 'red4num', 'red5num', 'red6num', 'bluenum-1',
            'red1num-2', 'red2num-2', 'red3num-2', 'red4num-2', 'red5num-2', 'red6num-2', 'bluenum-2',
            'red1num-3', 'red2num-3', 'red3num-3', 'red4num-3', 'red5num-3', 'red6num-3', 'bluenum-3',
            'red1num-4', 'red2num-4', 'red3num-4', 'red4num-4', 'red5num-4', 'red6num-4', 'bluenum-4',
            'red1num-5', 'red2num-5', 'red3num-5', 'red4num-5', 'red5num-5', 'red6num-5', 'bluenum-5']

train_data, test_data = train_test_split(bluenum_features, test_size=0.3)
train_data = train_data.copy()
test_data = test_data.copy()
train_sample = train_data[features]
train_result = train_data['bluenum']
test_sample = test_data[features]
test_result = test_data['bluenum']

# Train the Model of LR
print ('\n# Train the Model of LR...')
lr = LogisticRegression(penalty='l2', C=10000)
lr.fit(train_sample, train_result)
# print(lr.coef_)
# print(lr.intercept_)
# print(json.dumps(dict(zip(features, lr.coef_[0])), sort_keys=True))

# Predict
print ('\n# Predict...')
train_predict = lr.predict_proba(train_sample)
train_data['score'] = map(lambda i: max(i), train_predict)
train_data['predict'] = map(lambda i: np.where(i == max(i))[0][0], train_predict)
train_acc = train_data.groupby(by='dist')['dist'].count() / len(train_data)

test_predict = lr.predict_proba(test_sample)
test_data['score'] = map(lambda i: max(i), test_predict)
test_data['predict'] = map(lambda i: np.where(i == max(i))[0][0], test_predict)
test_acc = test_data.groupby(by='dist')['dist'].count() / len(test_data)


# Output Model
print ('\n# Output Model...')
print('\n')
coef_dict = dict(zip(features, lr.coef_[0]))
for fn in features:
    print fn, coef_dict[fn]
