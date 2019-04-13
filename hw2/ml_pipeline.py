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

# READ DATA
def load_csv(filename):
    '''This function reads a csv and returns a dataframe
    
    filename: csv file to read

    return: pandas dataframe
    '''
    return pd.read_csv(filename)

def load_xls(filename, sheetnum = 0):
    '''This function reads an xls and returns a dataframe
    
    filename: xls file to read
    sheetnum: sheet number to read, default is first at 0

    return: pandas dataframe
    '''
    return pd.read_excel(filename, sheetnum = sheetnum)


# EXPLORE DATA
def mk_hist_facet(df, tgt_col = []):
    '''
    This function plots a histogram for each numeric column in a dataframe.

    df: dataframe to plot
    tgt_col: list of columns to filter dataframe if not all columns need to be plotted (default is [], all columns plotted)

    return: none
    '''

    if tgt_col:
        df = df[tgt_col]

    df.hist()

def mk_corr_heatmap(df, tgt_col = [], corr_method = 'pearson', cscheme = 'coolwarm'):
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
    corr.style.background_gradient(cmap=scheme)

def find_outliers(df, tgt_col = [], thresh = 3):
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

    return df[(np.abs(sp.stats.zscore(df)) > thresh).any(axis=1)]


def rm_outliers(df, tgt_col = [], thresh = 3):
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

    return df[(np.abs(sp.stats.zscore(df)) < thresh).all(axis=1)]

def mk_summary(df, tgt_col = [], stat_types = ['count', 'mean', 'std', 'min', 'max'])
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
    This function fills NA values in a df by given method

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