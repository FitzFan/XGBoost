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

import hyperopt
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe


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

def score(params):
    logging.info("Training with params: ")
    logging.info(params)
    # Delete 'n_estimators' because it's only a constructor param
    # when you're using  XGB's sklearn API.
    # Instead, we have to save 'n_estimators' (# of boosting rounds)
    # to xgb.cv().
    num_boost_round = int(params['n_estimators'])
    del params['n_estimators']
    dtrain = xgb.DMatrix(X_train, label=y_train)
    # As of version 0.6, XGBoost returns a dataframe of the following form:
    # boosting iter | mean_test_err | mean_test_std | mean_train_err | mean_train_std
    # boost iter 1 mean_test_iter1 | mean_test_std1 | ... | ...
    # boost iter 2 mean_test_iter2 | mean_test_std2 | ... | ...
    # ...
    # boost iter n_estimators

    score_history = xgb.cv(params, dtrain, num_boost_round,
                           nfold=5, stratified=True,
                           early_stopping_rounds=250,
                           verbose_eval=500)
    # Only use scores from the final boosting round since that's the one
    # that performed the best.
    mean_final_round = score_history.tail(1).iloc[0, 0]
    std_final_round = score_history.tail(1).iloc[0, 1]
    logging.info("\tMean Score: {0}\n".format(mean_final_round))
    logging.info("\tStd Dev: {0}\n\n".format(std_final_round))
    # score() needs to return the loss (1 - score)
    # since optimize() should be finding the minimum, and AUC
    # naturally finds the maximum.
    loss = 1 - mean_final_round
    return {'loss': loss, 'status': STATUS_OK}

def optimize(random_state=SEED):
    """
    This is the optimization function that given a space (space here) of
    hyperparameters and a scoring function (score here),
    finds the best hyperparameters.
    """

    # space = {
    #     'n_estimators': hp.choice('n_estimators', [1000, 1100]),
    #     'eta': hp.quniform('eta', 0.01, 0.1, 0.025),
    #     'max_depth': hp.choice('max_depth', [4, 5, 7, 9, 17]),
    #     'min_child_weight': hp.choice('min_child_weight', [3, 5, 7]),
    #     'subsample': hp.choice('subsample', [0.4, 0.6, 0.8]),
    #     'gamma': hp.choice('gamma', [0.3, 0.4]),
    #     'colsample_bytree': hp.quniform('colsample_bytree', 0.4, 0.7, 0.1),
    #     'lambda': hp.choice('lambda', [0.01, 0.1, 0.9, 1.0]),
    #     'alpha': hp.choice('alpha', [0, 0.1, 0.5, 1.0]),
    #     'eval_metric': 'auc',
    #     'objective': 'binary:logistic',
    #     # Increase this number if you have more cores.
    #     # Otherwise, remove it and it will default
    #     # to the maxium number.
    #     'nthread': 4,
    #     'booster': 'gbtree',
    #     'tree_method': 'exact',
    #     'silent': 1,
    #     'seed': random_state
    # }
    space = {
        'n_estimators': hp.choice('n_estimators', [1000]),
        'eta': hp.choice('eta', [0.01]),
        'max_depth': hp.choice('max_depth', [4]),
        'min_child_weight': hp.choice('min_child_weight', [5]),
        'subsample': hp.choice('subsample', [0.4]),
        'gamma': hp.choice('gamma', [0.4, 0.8]),
        'colsample_bytree': hp.choice('colsample_bytree', [0.4]),
        'lambda': hp.choice('lambda', [0.9, 0.93]),
        'alpha': hp.choice('alpha', [0.5]),
        'eval_metric': 'auc',
        'objective': 'binary:logistic',
        # Increase this number if you have more cores.
        # Otherwise, remove it and it will default
        # to the maxium number.
        'nthread': 4,
        'booster': 'gbtree',
        'tree_method': 'exact',
        'silent': 1,
        'seed': random_state
    }
    # Use the fmin function from Hyperopt to find the best hyperparameters
    best = fmin(score, space, algo=tpe.suggest,  max_evals=4)
    return best

def main():
    # Let OpenMP use 4 threads to evaluate models - may run into errors
    # if this is not set. Should be set before hyperopt import.
    os.environ['OMP_NUM_THREADS'] = '4'

    SEED = 42  # Fix the random state to the ultimate answer in life.
    # Initialize logger
    logging.basicConfig(filename="logs/xgb_hyperopt.log", level=logging.INFO)

    # 读取数据
    id_train, X_train, y_train, id_test, X_test = load_data()
    X_train, X_test = preprocess_data(X_train, X_test)
    X_train, X_test = drop_features(X_train, X_test)

    best_hyperparams = optimize()
    print("The best hyperparameters are: ", "\n")
    print(best_hyperparams)

if __name__ == '__main__':
    main()

