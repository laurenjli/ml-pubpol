import ml_pipeline as pp
import pandas as pd
import datetime as dt
import numpy as np
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, BaggingClassifier, RandomForestClassifier, ExtraTreesClassifier
from sklearn.naive_bayes import GaussianNB
import ml_pipeline as pp

if __name__ == '__main__':

	file = './data/projects_2012_2013.csv'
	df = pp.load_csv(file)
	df.date_posted = pp.col_datetime(df, 'date_posted')
	df.datefullyfunded = pp.col_datetime(df,'datefullyfunded')
	df = pp.create_label(df, pred_time=60)
	windows = [dt.datetime(2012,1,1), dt.datetime(2012,7,1), dt.datetime(2013,1,1), dt.datetime(2013,7,1), dt.datetime(2014,1,1)]
	pred_time = 60 #days
	label_col = 'label'
	split_col = 'date_posted'
	feature_cols= [x for x in df.columns if x not in ['projectid', 'teacher_acctid', 'schoolid', 'school_ncesid', 'school_latitude', 'school_longitude', 'date_posted', 'datefullyfunded', 'label']]
	impute_info = ((pp.most_freq, ['school_metro','primary_focus_subject','primary_focus_area','resource_type','grade_level']), (np.nanmedian,['students_reached']))
	bucketdict= {'total_price_including_optional_support': (4, ('lowest', 'low', 'medium', 'high')), 'students_reached':(4, ('lowest', 'low', 'medium', 'high'))}
	top_k = {'school_state':30, 'school_city':30,'school_district':30, 'school_county':30}
	seed=12345

	models = [
    {'type': 'Dtree', 'clf': DecisionTreeClassifier(), 'params': {'criterion': ['entropy', 'gini'], 'max_depth': [10,20,30],'min_samples_split': [100, 300,500], 'random_state': [seed]}},
    {'type': 'LR', 'clf': LogisticRegression(), 'params':{'penalty': ['l1','l2'], 'C': [0.1, 1.0, 10.0, 100.0], 'solver': ['liblinear'], 'random_state': [seed]}},
    {'type': 'SVM', 'clf': LinearSVC(), 'params':{'penalty': ['l2'], 'C': [0.1, 1.0, 10.0, 100.0], 'random_state': [seed]}},
    {'type': 'Bagging_dtree', 'clf': BaggingClassifier(), 'params':{'n_estimators': [100, 500, 1000], 'base_estimator':[None], 'random_state':[seed]}},
    {'type': 'ADABoost_dtree', 'clf': AdaBoostClassifier(), 'params':{'n_estimators': [100, 500, 1000], 'base_estimator':[None], 'random_state':[seed]}},
    {'type': 'GBoost', 'clf': GradientBoostingClassifier(), 'params': {'n_estimators': [100, 500, 1000], 'min_samples_split': [100, 300,500], 'random_state':[seed]}},
    {'type': 'ExtraTrees', 'clf': ExtraTreesClassifier(),'params': {'n_estimators': [100, 500, 1000], 'criterion': ['entropy', 'gini'],'min_samples_split': [100, 300,500], 'max_depth': [10,20,30],'random_state':[seed], 'n_jobs':[5]}},
    {'type': 'Random Forest', 'clf': RandomForestClassifier(), 'params':{'n_estimators': [100, 500, 1000], 'criterion': ['entropy', 'gini'], 'random_state': [seed]}},
    {'type': 'KNN', 'clf': KNeighborsClassifier(), 'params':{'n_neighbors': [5,7], 'weights': ['uniform','distance'], 'metric':['minkowski'],'p': [1,2], 'n_jobs': [4]}},
    {'type': 'NB', 'clf': GaussianNB(),'params':{'priors':[None]}}
	]

	#models = [{'type': 'Dtree', 'clf': DecisionTreeClassifier(), 'params': {'criterion': ['entropy', 'gini'], 'max_depth': [10,20,30],'min_samples_split': [100, 300,500], 'random_state': [seed]}}]
	thresholds = [1, 2, 5, 10, 20,30, 50]

	pp.run_models(models, thresholds, windows, df, feature_cols, label_col, split_col, impute_info, bucketdict, top_k, pred_time, pred_unit = 'day', filename = './data/finalrun.csv')

