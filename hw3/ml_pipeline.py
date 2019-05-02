#Lauren Li
#
#This file contains the components of the machine learning pipeline.
#References: https://www.datacamp.com/community/tutorials/decision-tree-classification-python

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import datetime as dt
from dateutil.relativedelta import *
import scipy as sp
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO  
from IPython.display import Image  
#import pydotplus

# READ DATA
def load_csv(filename):
    '''This function reads a csv and returns a dataframe
    
    filename: csv file to read

    return: pandas dataframe
    '''
    return pd.read_csv(filename)

def load_xls(filename, sheetnum = 0, header = 1):
    '''This function reads an xls and returns a dataframe
    
    filename: xls file to read
    sheetnum: sheet number to read, default is first at 0
    header: which row to use as header

    return: pandas dataframe
    '''
    return pd.read_excel(filename, sheetnum = sheetnum, header = header)


# EXPLORE DATA
def mk_hist_facet(df, tgt_col = [], size= (16,12)):
    '''
    This function plots a histogram for each numeric column in a dataframe.

    df: dataframe to plot
    tgt_col: list of columns to filter dataframe if not all columns need to be plotted (default is [], all columns plotted)
    size = size of figure, default is (8, 6)
    return: none
    '''
    
    if tgt_col:
        df = df[tgt_col]
    
    plt.rcParams['figure.figsize'] = size
    df.hist()
    
    

def mk_corr_heatmap(df, tgt_col = [], corr_method = 'pearson'):
    '''
    This function creates a correlation heatmap from a dataframe.

    df: dataframe
    tgt_col: list of columns to filter dataframe if not all columns need to be plotted (default is [], all columns shown)
    corr_method: method to calculate correlation, default is pearson
    cscheme: color scheme for heatmap, default is 'coolwarm'

    return: none
    references: https://stackoverflow.com/questions/29432629/plot-correlation-matrix-using-pandas/31384328#31384328
    '''

    if tgt_col:
        df = df[tgt_col]

    corr = df.corr(method = corr_method)
    
    return corr

def find_outliers(df, tgt_col = [], thresh = 3, ignorena=False):
    '''
    This function finds the outliers according to a given threshold.

    df: dataframe
    tgt_col: list of columns to filter dataframe if not all columns need to be plotted (default is [], all columns shown)
    thresh: zscore threshold above which a data point is considered an outlier

    return: dataframe with outliers
    references: https://stackoverflow.com/questions/23199796/detect-and-exclude-outliers-in-pandas-data-frame
    '''

    if tgt_col:
        df = df[tgt_col]
    
    if ignorena:
        df.dropna(inplace = True)
        
    if np.isnan(np.abs(sp.stats.zscore(df))).any():
        print('WARNING: NA VALUES EXIST. You can choose to ignore NA values by including ignorena=True as an input to the function.')

    return df[(np.abs(sp.stats.zscore(df)) > thresh).any(axis=1)]


def rm_outliers(df, tgt_col = [], thresh = 3, ignorena=False):
    '''
    This function removes the outliers according to a given threshold.

    df: dataframe
    tgt_col: list of columns to filter dataframe if not all columns need to be plotted (default is [], all columns shown)
    thresh: zscore threshold above which a data point is considered an outlier

    return: dataframe with outliers
    references: https://stackoverflow.com/questions/23199796/detect-and-exclude-outliers-in-pandas-data-frame
    '''

    if tgt_col:
        df = df[tgt_col]
    
    if ignorena:
        df.dropna(inplace = True)
        
    if np.isnan(np.abs(sp.stats.zscore(df))).any():
        print('WARNING: NA VALUES EXIST. You can choose to ignore NA values by including ignorena=True as an input to the function.')

    return df[(np.abs(sp.stats.zscore(df)) < thresh).all(axis=1)]


def mk_summary(df, tgt_col = [], stat_types = ['count', 'mean', 'std', 'min', 'max']):
    '''
    This function returns a summary table for a dataframe

    df: dataframe
    tgt_col: list of columns to filter dataframe if not all columns need to be plotted (default is [], all columns shown)
    stat_types: summary stats to include, default is count, mean, std, min, max

    return: dataframe with categories as indices and stats as colnames
    references: http://social-metrics.org/summary-statistics-in-pandas/
    '''

    if tgt_col:
        df = df[tgt_col]

    return np.round(df.describe(), 2).T[stat_types]


# PRE-PROCESS DATA
def col_datetime(df, tgt_col):
    '''
    This function creates a column of datetimes from a given column

    df: dataframe
    tgt_col: column name

    returns: pandas series with datetimes
    '''
    return pd.to_datetime(df[tgt_col])

def na_exists(df, colname):
    '''
    This function returns a boolean for whether or not a pd series contains null values.
    
    df: dataframe
    colname: column of df to look at
    
    returns: True or False
    '''
    return df[colname].isnull().values.any()

def na_col(df):
    '''
    This function identifies which columns in a dataframe have NA values
    
    df: dataframe
    
    returns: list of colnames with NA values
    '''
    
    return df.columns[df.isna().any()].tolist()

def most_freq(col_series):
    '''
    This function identifies the most frequent category in a column

    df: dataframe
    col_name: string column name

    returns: category with highest frequency
    '''

    return col_series.value_counts().keys()[0]

def na_fill_all(df, fill_method = np.mean):
    '''
    This function fills NA values in a df by given method

    df: dataframe
    fill_method: function specifying how to fill the NA values

    return: dataframe with NA values filled
    '''
    return df.fillna(fill_method(df))

def na_fill_col(df, col, fill_method = np.mean):
    '''
    This function fills NA values in a df column by given method

    df: dataframe
    fill_method: function specifying how to fill the NA values
    col: column name

    return: None
    '''
    df.loc[df[col].isna(), col] = fill_method(df[col])
    return df
    

def missing_fill(df, col, missing_id, fill_method = np.mean):
    '''
    This function fills values in a df identified by a specific trait (i.e. 999999) by given method

    df: dataframe
    col: column with missing value
    missing_id: identifier for missing value (i.e. in case its not NA)
    fill_method: function specifying how to fill the NA values

    return: dataframe with missing values filled
    '''

    df.loc[df[col == missing_id], col] = np.mean(df[col])
    return df



def subset_dropna(df, tgt_col, how_type = 'all'):
    '''
    This function drops na's if there are na's in the columns specified by subset

    df: dataframe
    tgt_col: list of column names to look for NA's
    how_type: how to drop NA's, default is 'all' i.e. there needs to be NA in every column in tgt_col to drop
    
    return: dataframe without NA values
    '''

    return df.dropna(subset= tgt_col, how= how_type)


# Generate features

def feat_binary(df, tgt_col = []):
    '''
    This function replaces the categorical variables with one-hot representations.
    
    df: dataframe
    tgt_col: list of colnames to make binary
    
    return: full dataframe with dummy variables in new columns
    '''
    
    return pd.get_dummies(df, columns = tgt_col)


def feat_sing_disc(df, colname, bucket = 3):
    '''
    This function discretizes a column with continuous variables.
    
    df: dataframe
    colname: column name
    bucket: int with number of buckets or list of floats to use as range
    labels: list of labels for ranges if want string to show up instead of interval
    
    return: df with interval intead of continuous variable
    '''
    colname2 = colname + '_binned'
    df[colname2] = pd.cut(df[colname], bucket)
    
    return df


def feat_mult_disc(df, bucketdict, qt = False):
    '''
    This function discretizes multiple columns with continuous variables.
    
    df: dataframe
    bucketdict: dictionary containing colnames as keys and bucket int/list as values for range
    
    return: df with interval intead of continuous variable
    '''
    
    for k, v in bucketdict.items():
        if qt:
            df = feat_sing_qt(df, colname = k, qlist = v)
        else:
            df = feat_sing_disc(df, colname = k, bucket = v)
        
    return df.drop(list(bucketdict.keys()), axis=1)
    

def feat_sing_qt(df, colname, qlist = [0, .25, .5, .75, 1.], duplicates='drop'):
    '''
    This function discretizes a column with continuous variables into buckets based on quantiles.
    
    df: dataframe
    colname: column name
    qlist: range of quantiles to use
    labels: list of labels for ranges if want string to show up instead of interval
    
    return: df with interval intead of continuous variable
    '''
    colname2 = colname + '_binned'
    df[colname2] = pd.qcut(df[colname], q = qlist, duplicates = 'drop')
    
    return df


def create_label(df, pred_time = 60, pred_unit = 'day'):
    '''
    This function creates a label column in the dataset given a time horizon
    
    df: dataframe
    pred_time: prediction time horizon
    pred_unit: unit of time of prediction (i.e. day or month or year)
    
    return dataframe
    '''
    def label(row, pred_time, pred_unit):
        if 'day' in pred_unit:
            diff = pred_time
        elif 'month' in pred_unit:
            diff = pred_time * 30 #convert to days
        elif 'year' in pred_unit:
            diff = pred_time * 365 #convert to days
            
        if (row.datefullyfunded - row.date_posted).days <= diff:
            return 1
        else:
            return 0

    df['label'] = df.apply(lambda row: label(row, pred_time, pred_unit), axis=1)
    
    return df


# Temporal validation

def build_windows(first_date, last_date, split_time, split_type = 'month'):
    '''
    This function creates a list of dates to split data on.
    
    first_date: date to begin on in data
    last_date: date to end on in data
    split_time: number of days/months/years to split data
    split_type: day/month/year
    
    return: list of datetimes
    '''
    windows = [first_date]
    
    if split_type == 'day':
        tmp = first_date
        while tmp < last_date:
            tmp += dt.timedelta(days =split_time)
            windows.append(tmp)

    elif split_type == 'month':
        tmp = first_date
        while tmp < last_date:
            tmp += relativedelta(months = +split_time)
            windows.append(tmp)

    elif split_type == 'year':
        tmp = first_date
        while tmp < last_date:
            tmp = dt.datetime(tmp.year + split_time, tmp.month, tmp.day)
            windows.append(tmp)

    return windows

def single_train_test_set(df, feature_cols, label_col, split_col, train_start, train_end, test_end, pred_time, pred_unit = 'day'):
    '''
    This function builds a single temporal training and test set
    
    df: dataframe with all data
    feature_cols: list of feature columns
    label_col: label column string
    split_col: column with date to split on
    train_start: date to begin training data
    train_end: date to end training data
    test_end: date to end test data
    pred_time: time horizon for predictions (day, month or year)
    
    returns tuple of x_train, y_train, x_test, y_test
    '''
    
    if 'day' in pred_unit:
        actual_train_end = train_end - dt.timedelta(days=pred_time)
    elif 'month' in pred_unit:
        actual_train_end = train_end - relativedelta(months = +pred_time)
    elif 'year' in pred_unit.contains:
        actual_train_end = dt.datetime(train_end.year - pred_time, train_end.month, train_end.day)
        
    training_set = df[(df[split_col] >= train_start) & (df[split_col] <= actual_train_end)]
    
    test_set = df[(df[split_col] >= train_end) & (df[split_col] <= test_end)]
    
    return (training_set[feature_cols], training_set[label_col], test_set[feature_cols], test_set[label_col])



# Build Classifiers
# referenced: https://www.datacamp.com/community/tutorials/decision-tree-classification-python


def dtree_score(feature_train, label_train, feature_test, criteria = 'entropy', depth = 5, min_leaf = 500, seed = 12345):
    '''
    This function builds a decision tree using training data

    feature_train: feature set in training data
    label_train: labels in training data
    criteria: how to split (entropy or gini)
    depth: maximum depth of tree

    return: trained decision tree classifier
    '''
    tree = DecisionTreeClassifier(criterion= criteria, max_depth= depth, min_samples_leaf = min_leaf, random_state= seed)

    tree.fit(feature_train, label_train)

    return tree.predict_proba(feature_test)[:,1]

def knn_score(x_train, y_train, x_test, n, weights = 'uniform', distance_metric = 'minkowski', p=2):
    '''
    This function builds a KNN classifier.
    
    x_train: training set with features
    y_train: training set with labels
    n: number of neighbors
    weights: weight function used in prediction.
    distance_metric: the distance metric to use
    p: Power parameter for the Minkowski metric. 
    
    returns: fitted KNN classifier
    '''
    if distance_metric == 'minkowski':
        knn = KNeighborsClassifier(n_neighbors=n, weights = weights, p=p, metric= distance_metric)
        
    else:
        knn =KNeighborsClassifier(n_neighbors=n, weights = weights, metric= distance_metric)
    
    knn.fit(x_train, y_train)
    
    return predictpr(knn, x_test)

def lr_score(x_train, y_train, x_test, p = 'l1', c = 1.0, solver = 'liblinear', seed=12345):
    '''
    This function builds a Logistic Regression model.
    
    x_train: training set with features
    y_train: training set with labels
    p: penalty (l1 or l2)
    c: Inverse of regularization strength; must be a positive float.
    solver: Algorithm to use in the optimization problem.
    seed: random seed
    
    returns fitted Logistic Regression
    '''
    
    lr = LogisticRegression(penalty = p, C = c, solver= solver, random_state = seed)
    lr.fit(x_train, y_train)
    
    return predictpr(lr, x_test)

def linsvc_score(x_train, y_train, x_test, p = 'l2', c = 1.0, seed = 12345):
    '''
    This function builds a fitted linear SVC
    
    x_train: training set with features
    y_train: training set with labels
    p: penalty (l2)
    c: Penalty parameter C of the error term
    seed: random seed
    
    returns fitted linear SVC
    '''
    
    lsvc = LinearSVC(penalty = p, C=c, random_state=seed)
    lsvc.fit(x_train, y_train)
    
    return lsvc.decision_function(x_test)

def adaboost_score(x_train, y_train, x_test, n, base = None, seed=12345):
    '''
    This function creates and predicts scores using bagging.
    
    x_train: training set features
    y_train: training set labels
    x_test: test set features
    n_estimators: number of estimators to be in bagging
    base: base classifier, default None is Dtree
    n_jobs = jobs to do in parallel
    
    returns: predicted scores 
    '''
    
    ada = AdaBoostClassifier(base_estimator = base, n_estimators=n, random_state=seed)
    ada.fit(x_train, y_train)
    return predictpr(ada,x_test)

def bagging_score(x_train, y_train, x_test, n, base = None, n_jobs = 1, seed= 12345):
    '''
    This function creates and predicts scores using bagging.
    
    x_train: training set features
    y_train: training set labels
    x_test: test set features
    n_estimators: number of estimators to be in bagging
    base: base classifier, default None is Dtree
    n_jobs = jobs to do in parallel
    
    returns: predicted scores 
    '''
    
    bag = BaggingClassifier(base_estimator=base, n_estimators= n, n_jobs = n_jobs, random_state = seed)
    bag.fit(x_train, y_train)
    return predictpr(bag, x_test)

def rforest_score(x_train, y_train, x_test, n, criterion = 'entropy', max_depth = None, n_jobs= None, seed=seed):
    '''
    This function returns probabilities from a Random Forest Classifier
    
    x_train: training set features
    y_train: training set labels
    x_test: test set features
    n: n_estimators
    criterion: entropy or gig
    max_depth: depth of tree, None default means nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.
    n_jobs: number of jobs to run in parallel for both fit and predict. None means 1
    seed: random seed
    
    return: prediction scores
    '''
    
    rf = RandomForestClassifier(n_estimators=n, criterion= criterion, max_depth=max_depth, n_jobs=n_jobs, random_state=seed)
    rf.fit(x_train, y_train)
    
    return predictpr(rf, x_test)
    
def predictpr(fitted, feature_test):
    '''
    This function predicts the probability of response 1 for the test dataset.
    
    fitted: fitted classifier
    feature_test: feature set in test data
    
    return: predictions
    '''
    return fitted.predict_proba(feature_test)[:,1]


#Run multiple classifiers and save in dataframe

def run_models(models, thresholds, windows, df_final, feature_cols, label_col, split_col, pred_time, pred_unit = 'day', filename = ''):
    '''
    This function runs multiple models with multiple parameters and calculates metrics according to thresholds

    models: list of dictionaries, each one is a model type with parameters
    thresholds: list of thresholds to calculate metrics against for each model
    windows: list of start and end dates for time windows
    feature_cols: list of strings, column names
    label_col: column name of label
    split_col: column name of column that has the dates to split on
    pred_time: prediction window
    pred_unit: time unit for prediction window
    filename: csv filename to save results

    returns: dataframe
    '''

    results = []

    for i in range(1, len(windows)-1):
        train_start = windows[0]
        train_end = windows[i]
        test_end = windows[i+1]
        
        #split data
        x_train,y_train,x_test,y_test = pp.single_train_test_set(df_final, 
                                                            feature_cols, 
                                                            label_col, 
                                                            split_col, 
                                                            train_start,
                                                            train_end, 
                                                            test_end, 
                                                            pred_time=pred_time, pred_unit = pred_unit)
        
        
        baseline = sum(y_test)/len(y_test)
        #run models
        for clf in models:
            modeltype = clf['type']
            func = clf['clf']
            printinfo = 'model: {}, run: {}'.format(modeltype, i)
            print(printinfo)
            if modeltype == 'Dtree':
                seed = clf['seed']
                for c in clf['criteria']:
                    for d in clf['depth']:
                        for l in clf['min_leaf']:
                            #run model
                            info = 'criteria: {}, depth: {}, min_leaf: {}, seed: {}'.format(c, d, l, seed)
                            print(info)
                            scores = func(x_train, y_train, x_test, criteria = c, depth = d, min_leaf = l, seed=seed)
                            for pct_pop in thresholds:
                                acc, prec, rec, f1, auc = all_metrics(y_test, scores, pct_pop)
                                tmp = {'baseline': baseline, 'train_set_num': i, 'train_start': train_start, 'test_start': train_end,'type': modeltype, 
                                    'details': info, 'threshold_pct': pct_pop, 'precision': prec, 'recall': rec, 'auc': auc}
                                results.append(tmp)
                                
            elif modeltype == 'LR':
                seed = clf['seed']
                for p in clf['p']:
                    for c in clf['c']:
                        for s in clf['solver']:
                            #print(p)
                            info = 'penalty: {}, c: {}, solver: {}, seed: {}'.format(p, c, s, seed)
                            print(info)
                            scores = func(x_train, y_train, x_test, p = p, c = c, solver = s, seed=seed)
                            for pct_pop in thresholds:
                                acc, prec, rec, f1, auc = all_metrics(y_test, scores, pct_pop)
                                tmp = {'baseline': baseline,'train_set_num': i, 'train_start': train_start, 'test_start': train_end,'type': modeltype, 
                                    'details': info, 'threshold_pct': pct_pop, 'precision': prec, 'recall': rec, 'auc': auc}
                                results.append(tmp)
                            
            elif modeltype == 'SVM':
                seed = clf['seed']
                for p in clf['p']:
                    for c in clf['c']:
                        info = 'penalty: {}, c: {}, seed: {}'.format(p, c, seed)
                        print(info)
                        scores = func(x_train, y_train, x_test, p =p, c=c, seed=seed)
                        for pct_pop in thresholds:
                            acc, prec, rec, f1, auc = all_metrics(y_test, scores, pct_pop)
                            tmp = {'baseline': baseline,'train_set_num': i, 'train_start': train_start, 'test_start': train_end,'type': modeltype, 
                                'details': info, 'threshold_pct': pct_pop, 'precision': prec, 'recall': rec, 'auc': auc}
                            results.append(tmp)
                                
            elif modeltype == 'KNN':
                for n in clf['n']:
                    for w in clf['weights']:
                        for d in clf['distance_metric']:
                            for p in clf['p']:
                                info = 'n: {}, weights: {}, distance: {}, p: {}'.format(n, w, d, p)
                                print(info)
                                scores = func(x_train, y_train, x_test, n = n, weights = w, distance_metric = d, p=p)
                                #print(list(scores))
                                for pct_pop in thresholds:
                                    acc, prec, rec, f1, auc = all_metrics(y_test, scores, pct_pop)
                                    tmp = {'baseline': baseline,'train_set_num': i, 'train_start': train_start, 'test_start': train_end,'type': modeltype, 
                                        'details': info, 'threshold_pct': pct_pop, 'precision': prec, 'recall': rec, 'auc': auc}
                                    results.append(tmp)
            
            elif modeltype == 'Bagging_dtree':
                seed = clf['seed']
                for n in clf['n']:
                    for b in clf['base']:
                        info = 'n: {}, base: {}'.format(n,b)
                        scores = func(x_train, y_train, x_test, n = n, base = b, seed=seed)
                        for pct_pop in thresholds:
                            acc, prec, rec, f1, auc = all_metrics(y_test, scores, pct_pop)
                            tmp = {'baseline': baseline,'train_set_num': i, 'train_start': train_start, 'test_start': train_end,'type': modeltype, 
                                'details': info, 'threshold_pct': pct_pop, 'precision': prec, 'recall': rec, 'auc': auc}
                            results.append(tmp)
            
            elif modeltype == 'ADABoost_dtree':
                seed = clf['seed']
                for n in clf['n']:
                    for b in clf['base']:
                        info = 'n: {}, base: {}'.format(n,b)
                        scores = func(x_train, y_train, x_test, n = n, base = b, seed=seed)
                        for pct_pop in thresholds:
                            acc, prec, rec, f1, auc = all_metrics(y_test, scores, pct_pop)
                            tmp = {'baseline': baseline,'train_set_num': i, 'train_start': train_start, 'test_start': train_end,'type': modeltype, 
                                'details': info, 'threshold_pct': pct_pop, 'precision': prec, 'recall': rec, 'auc': auc}
                            results.append(tmp)
                
                
            elif modeltype == 'Random Forest':
                seed = clf['seed']
                for n in clf['n']:
                    for c in clf['criterion']:
                        info = 'n: {}, criterion: {}'.format(n,c)
                        scores = func(x_train, y_train, x_test, n = n, criterion = c, seed=seed)
                        for pct_pop in thresholds:
                            acc, prec, rec, f1, auc = all_metrics(y_test, scores, pct_pop)
                            tmp = {'baseline': baseline,'train_set_num': i, 'train_start': train_start, 'test_start': train_end,'type': modeltype, 
                                'details': info, 'threshold_pct': pct_pop, 'precision': prec, 'recall': rec, 'auc': auc}
                            results.append(tmp)
                                      
    resdf = pd.DataFrame(results, columns = ['type', 'details', 'baseline', 'threshold_pct', 'precision', 'recall', 'auc','train_set_num', 'train_start', 'test_start'])    
    if filename:
        resdf.to_csv(filename)
    return resdf

# Visualize tree

# def graph_tree(tree, feature_list, filename):
#     dot_data = StringIO()
#     export_graphviz(tree, out_file=dot_data,  
#                     filled=True, rounded=True,
#                     special_characters=True,feature_names = feature_list,class_names=['0','1'])
#     graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
#     graph.write_png(filename)

    
# Evaluate

def accuracy_at_threshold(y_test, pred_scores, thresh =0.5):
    '''
    This function calculates that accuracy of model given a threshold

    y_test: real labels
    pred: prediction scores
    threshold: threshold for predictions

    returns accuracy 
    '''
    pred_one = [1 if x > thresh else 0 for x in pred_scores]
    
    return metrics.accuracy_score(y_test, pred_one)

def build_cmatrix(y_test, pred_scores, threshold):
    '''
    This function builds a confusion matrix for a given threshold
    
    pred_scores: prediction scores
    y_test: real labels
    threshold: threshold for predictions
    
    returns tuple (true_negatives, false_positive, false_negatives, true_positives)
    '''
    pred = [1 if x > threshold else 0 for x in pred_scores]
    
    cmatrix = confusion_matrix(y_test, pred)
    
    true_negatives, false_positive, false_negatives, true_positives = cmatrix.ravel()

    return (true_negatives, false_positive, false_negatives, true_positives)

def precision_at_threshold(y_test, pred_scores, thresh =0.5):
    '''
    This function calculates precision of model given a threshold

    y_test: real labels
    pred_scores: prediction scores
    threshold: threshold for predictions

    returns precision
    '''
    pred_one = [1 if x > thresh else 0 for x in pred_scores]
    
    return metrics.precision_score(y_test, pred_one)

def recall_at_threshold(y_test, pred_scores, thresh =0.5):
    '''
    This function calculates recall of model given a threshold

    y_test: real labels
    pred_scores: prediction scores
    threshold: threshold for predictions

    returns recall
    '''
    pred_one = [1 if x > thresh else 0 for x in pred_scores]
    
    return metrics.recall_score(y_test, pred_one)

def f1_at_threshold(y_test, pred_scores, thresh =0.5):
    '''
    This function calculates that accuracy of model given a threshold

    y_test: real labels
    pred_scores: prediction scores
    threshold: threshold for predictions

    returns f1 score 
    '''
    pred_one = [1 if x > thresh else 0 for x in pred_scores]
    
    return metrics.f1_score(y_test, pred_one)

def auc_roc(y_test, pred_scores):
    '''
    This function calculates the area under the ROC curve
    
    y_test: real labels
    pred_scores: prediction scores
    
    returns auc
    '''
    
    return metrics.roc_auc_score(y_test, pred_scores)

def scores_pctpop(pred_scores, pct_pop):
    
    #identify number of positives to have given target percent of population
    num_pos = int(round(len(pred_scores)*(pct_pop/100),0))
    #turn predictions into series
    pred_df = pd.Series(pred_scores)
    idx = pred_df.sort_values(ascending=False)[0:num_pos].index 
    
    #set all observations to 0
    pred_df.iloc[:] = 0
    #set observations by index (the ones ranked high enough) to 1
    pred_df.iloc[idx] = 1
    
    return pred_df

def all_metrics(y_test, pred_scores, t, target_pop = True):
    '''
    This function returns the accuracy, precision, recall, f1, and auc_roc for a given target percent of population
    
    y_test: tests set labels
    pred_scores: prediction scores
    t: threshold (either decimal as threshold or integer as % target population (50 is 50%))
    target_pop: boolean to decide whether to use t as threshold or target pop
    
    return: tuple with accuracy, precision, recall, f1, and auc_roc
    '''
    if target_pop:
        pred_scores = scores_pctpop(pred_scores, t)
        t = 0.5

    acc = accuracy_at_threshold(y_test, pred_scores, t)
    prec = precision_at_threshold(y_test, pred_scores, t)
    rec = recall_at_threshold(y_test, pred_scores, t)
    f1 = f1_at_threshold(y_test, pred_scores, t)
    auc = auc_roc(y_test, pred_scores)
    
    return (acc, prec, rec, f1, auc)

def plot_precision_recall(y_test, pred_scores):
    '''
    This function plots the precision recall curve
    
    y_test: true labels
    pred_scores: predicted scores
    
    return: none
    '''
    precision, recall, thresholds = metrics.precision_recall_curve(y_test, pred_scores)
    plt.plot(recall, precision, marker='.')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.show()

def plot_pct_pop(y_test, pred_scores):
    '''
    This function plots precision and recall on two axes with percent of population as the x-axis.
    
    y_test: test set labels
    pred_scores: predicted scores
    
    return: None
    '''
    pct_pop = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    prec = []
    rec = []
    for each in pct_pop:
        #quantile = 100 - each
        #t = np.percentile(pred_scores, quantile)
        a, p, r, f1, auc = all_metrics(y_test, final_p, each)
        #p = precision_at_threshold(y_test, pred_scores, t)
        #r = recall_at_threshold(y_test, pred_scores, t)
        prec.append(p)
        rec.append(r)
    
    #pct_pop = 1-thresholds
    fig, ax1 = plt.subplots()

    color = 'tab:blue'
    ax1.set_xlabel('percent of population')
    ax1.set_ylabel('precision', color=color)
    ax1.plot(pct_pop, prec, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:red'
    ax2.set_ylabel('recall', color=color)  # we already handled the x-label with ax1
    ax2.plot(pct_pop, rec, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()



if __name__ == '__main__':
    