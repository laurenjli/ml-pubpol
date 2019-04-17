#Lauren Li
#
#This file contains the components of the machine learning pipeline.
#References: https://www.datacamp.com/community/tutorials/decision-tree-classification-python

import pandas as pd 
import numpy as np 
import matplotlib as plt
import scipy as sp
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO  
from IPython.display import Image  
import pydotplus

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

def na_fill(df, fill_method = np.mean):
    '''
    This function fills NA values in a df by given method

    df: dataframe
    fill_method: function specifying how to fill the NA values

    return: dataframe with NA values filled
    '''
    return df.fillna(fill_method(df))


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
    
    df[colname] = pd.cut(df[colname], bucket)
    
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
        
    return df
    

def feat_sing_qt(df, colname, qlist = [0, .25, .5, .75, 1.], duplicates='drop'):
    '''
    This function discretizes a column with continuous variables into buckets based on quantiles.
    
    df: dataframe
    colname: column name
    qlist: range of quantiles to use
    labels: list of labels for ranges if want string to show up instead of interval
    
    return: df with interval intead of continuous variable
    '''
    
    df[colname] = pd.qcut(df[colname], q = qlist, duplicates = 'drop')
    
    return df


# Build Decision Tree
# referenced: https://www.datacamp.com/community/tutorials/decision-tree-classification-python

def split_data(df, feature_list, label_col, tsize = 0.25, seed = 12345):
    '''
    This function splits a dataframe into training and test data.
    
    df: dataframe
    feature_list: list of feature column names to extract from df
    label_col: name of label column to extract from df
    tsize: size of test set (default is 0.25 test, 0.75 training)
    seed: random seed generator
    
    return: iterable with arrays for the x_train (feature set in training data), y_train (labels in training data),
        x_test (feature set in test data), y_test (labels in test data)
    '''
    
    features = df[feature_list]
    labels = df[label_col]
    
    return train_test_split(features, labels, test_size=tsize, random_state=seed)

def build_dtree(feature_train, label_train, criteria = 'entropy', depth = 5, min_leaf = 500, seed = 12345):
    '''
    This function builds a decision tree using training data

    feature_train: feature set in training data
    label_train: labels in training data
    criteria: how to split (entropy or gini)
    depth: maximum depth of tree

    return: trained decision tree classifier
    '''
    tree = DecisionTreeClassifier(criterion= criteria, max_depth= depth, min_samples_leaf = min_leaf, random_state= seed)

    fitted = tree.fit(feature_train, label_train)

    return fitted
    
def predict_dtree(fitted_tree, feature_test):
    '''
    This function predicts the response for the test dataset.
    
    fitted_tree: fitted tree classifier
    feature_test: feature set in test data
    
    return: predictions
    '''
    return fitted_tree.predict(feature_test)

# Visualize

def graph_tree(tree, feature_list, filename):
    dot_data = StringIO()
    export_graphviz(tree, out_file=dot_data,  
                    filled=True, rounded=True,
                    special_characters=True,feature_names = feature_list,class_names=['0','1'])
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
    graph.write_png(filename)

    
# Evaluate

def accuracy(label_test, pred):
    '''
    This function finds the accuracy of predictions.
    
    label_test: labels in test data
    pred: predictions
    
    return: float
    '''
    
    return metrics.accuracy_score(label_test, pred)
        
        