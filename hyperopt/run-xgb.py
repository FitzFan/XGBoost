#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@Author  : Ryan Fan 
@E-Mail  : ryanfan0528@gmail.com
@Version : v1.0
"""

import os
import logging
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import Imputer
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.feature_selection import VarianceThreshold


def load_data():
    # Read data
    train = pd.read_csv('data/train_final.csv')
    test = pd.read_csv('data/test_final.csv')
    # Split row IDs off from features
    id_train = train.id
    id_test = test.id
    train = train.drop(['id'], axis=1)
    test = test.drop(['id'], axis=1)
    # Split dataset into features and target
    y_train = train.Y
    X_train = train.drop(["Y"], axis=1)

    return id_train, X_train, y_train, id_test, test

def preprocess_data(X_train, X_test):
    """ Impute missing values. """
    # Impute using the mean of every column for now. However,
    # I would've liked to impute 'F5' using mode instead.
    imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
    train_xform = imp.fit_transform(X_train)

    X_train = pd.DataFrame(train_xform, columns=X_train.columns)
    test_xform = imp.transform(X_test)
    X_test = pd.DataFrame(test_xform, columns=X_test.columns)

    return X_train, X_test

def drop_features(X_train, X_test):
    # Drop some features from the get-go. No idea how these were found.
    X_train = X_train.drop(['F6', 'F26'], 1)
    X_test = X_test.drop(['F6', 'F26'], 1)

    # Drop additional low-variance features. This *may* be overfitting to the
    # test data, since the hyperparameters are different for train/test.
    X_train = VarianceThreshold(1.3).fit_transform(X_train)
    X_test = VarianceThreshold(1.25).fit_transform(X_test)

    return X_train, X_test

def main():
    SEED = 42  # Fix the random state to the ultimate answer in life.
    # Initialize logger
    logging.basicConfig(filename="logs/xgb_hyperopt.log", level=logging.INFO)

    # 读入数据
    id_train, X_train, y_train, id_test, X_test = load_data()
    X_train, X_test = preprocess_data(X_train, X_test)
    X_train, X_test = drop_features(X_train, X_test)

    # Raymond Wen's parameters
    params = {
        'n_estimators': 1000,
        'eta': 0.01,
        'max_depth': 4,
        'min_child_weight': 5,
        'subsample': 0.4,
        'gamma': 0.8,
        'colsample_bytree': 0.4,
        'lambda': 0.93,
        'alpha': 0.5,
        'eval_metric': 'auc',
        'objective': 'binary:logistic',
        # Increase this number if you have more cores.
        # Otherwise, remove it and it will default
        # to the maxium number.
        'nthread': 4,
        'booster': 'gbtree',
        'tree_method': 'exact',
        'silent': 1,
        'seed': SEED
    }

    # Train the model

    num_boost_round = int(params['n_estimators'])
    del params['n_estimators']

    # 将数据格式转换为DMatrix对象
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test)
    # As of version 0.6, XGBoost returns a dataframe of the following form:
    # boosting iter | mean_test_err | mean_test_std | mean_train_err | mean_train_std
    # boost iter 1 mean_test_iter1 | mean_test_std1 | ... | ...
    # boost iter 2 mean_test_iter2 | mean_test_std2 | ... | ...
    # ...
    # boost iter n_estimators

    xg_booster = xgb.train(params, dtrain, num_boost_round)
    predictions = xg_booster.predict(dtest)
    # Predict class probabilities
    d = {'Id': id_test, 'Y': predictions}
    submit = pd.DataFrame(d)
    submit.to_csv('output/xgb_raymond_seed42.csv', index=False)

if __name__ == '__main__':
    main()

