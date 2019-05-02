import ml_pipeline as pp
import pandas as pd
import datetime as dt
import numpy as np

if __name__ == '__main__':
    df_final = pp.load_csv('./data/cleaned.csv')
    df_final.date_posted = pp.col_datetime(df_final, 'date_posted')

    windows = [dt.datetime(2012,1,1), dt.datetime(2012,7,1), dt.datetime(2013,1,1), dt.datetime(2013,7,1), dt.datetime(2014,1,1)]
    pred_time = 60 #days
    label_col = 'label'
    split_col = 'date_posted'
    feature_cols= list(df_final.columns)
    feature_cols.remove('label')
    feature_cols.remove('date_posted')
    seed=12345

    models = [
        {'type': 'Dtree', 'clf': pp.dtree_score, 'criteria': ['entropy', 'gini'], 'depth': [10,20,30],'min_leaf': [100, 300,500], 'seed': seed},
        {'type': 'LR', 'clf': pp.lr_score, 'p': ['l1','l2'], 'c': [0.1, 1.0, 10.0, 100.0], 'solver': ['liblinear'], 'seed': seed},
        {'type': 'SVM', 'clf': pp.linsvc_score, 'p': ['l2'], 'c': [0.1, 1.0, 10.0, 100.0], 'seed': seed},
        {'type': 'Bagging_dtree', 'clf': pp.bagging_score, 'n': [10, 50, 100], 'base':[None], 'seed':seed},
        {'type': 'ADABoost_dtree', 'clf': pp.adaboost_score, 'n': [10, 50, 100], 'base':[None], 'seed':seed},
        {'type': 'Random Forest', 'clf': pp.rforest_score, 'n': [10, 50, 100], 'criterion': ['entropy', 'gini'], 'seed': seed},
        {'type': 'KNN', 'clf': pp.knn_score, 'n': [5], 'weights': ['uniform','distance'], 'distance_metric':['minkowski'],'p': [1,2]}
    ]

    #models = [{'type': 'Random Forest', 'clf': rforest_score, 'n': [10, 50, 100], 'criterion': ['entropy', 'gini'], 'seed': seed}]
    thresholds = [1, 2, 5, 10, 20,30, 50]

    resdf = pp.run_models(models, thresholds, windows, df_final, feature_cols, label_col, split_col, pred_time, pred_unit = 'day', filename = './data/finalrun2.csv')

