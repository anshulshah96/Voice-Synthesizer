from pandas import DataFrame
import pandas as pd

import numpy as np

import sklearn
from sklearn import metrics
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import itertools

def get_partitions(df, group_by_column='fname'):
	df_train = pd.DataFrame()
	df_cross = pd.DataFrame()
	df_test = pd.DataFrame()
	data_percentage = 100 # in integral value

	df = df.groupby('pid')
	for name, group in df:
	    file_names = group[group_by_column].drop_duplicates()
	    file_names = file_names[:data_percentage]
	    file_train, file_test, _, _ = train_test_split(file_names, file_names, test_size=0.3, random_state=42)
	    file_test, file_cv, _, _ = train_test_split(file_test, file_test, test_size=0.5, random_state=42)
	    df_train = df_train.append(group[group[group_by_column].isin(file_train)])
	    df_cross = df_cross.append(group[group[group_by_column].isin(file_cv)])
	    df_test = df_test.append(group[group[group_by_column].isin(file_test)])
    	# print name, len(file_train), len(file_cv), len(file_test)
	return df_train, df_cross, df_test

def get_partitions_2(df, group_by_column='fname', train_per = 0.6, cross_per=0.2, test_per=0.2):
	df_train = pd.DataFrame()
	df_test = pd.DataFrame()
	df_cross = pd.DataFrame()
	data_percentage = 100 # in integral value

	df = df.groupby('pid')

	for name, group in df:
	    file_names = group[group_by_column].drop_duplicates()
	    file_names = file_names[:data_percentage]
	    file_train, file_test, _, _ = train_test_split(file_names, file_names, test_size=(1.0-train_per), random_state=42)

	    ratio_cross = cross_per/(cross_per+test_per)
	    file_test, file_cv, _, _ = train_test_split(file_test, file_test, test_size=ratio_cross, random_state=42)
	    df_train = df_train.append(group[group[group_by_column].isin(file_train)])
	    df_cross = df_cross.append(group[group[group_by_column].isin(file_cv)])
	    df_test = df_test.append(group[group[group_by_column].isin(file_test)])
    	# print name, len(file_train), len(file_cv), len(file_test)
	return df_train, df_cross, df_test

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This functi4on prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    # plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        # print("Normalized confusion matrix")
    # else:
        # print('Confusion matrix, without normalization')

    # print(cm)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')