import os
import sys

from models import *
import numpy as np
import pandas as pd
import matplotlib as plt
import sklearn
from numpy import zeros
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
from sklearn import datasets
from sklearn.externals import joblib

le = LabelEncoder()
df = pd.read_hdf("data_extract/features_dataset_2b.h5")
NUM_PEOPLE = 5
df = df.loc[df['id']<NUM_PEOPLE]
clist = ["chroma{}".format(i) for i in range(12)]
lclist = ["lc{}".format(i) for i in range(12)]
df[lclist] = np.log(df[clist])
flist = ["mfcc{}".format(i) for i in range(14)]+lclist+["centroid","crest","flatness","kurtosis","mean"]
df_train, df_cross, df_test = get_partitions(df)
obj = SVM(flist)
obj.fit(df_train)
joblib.dump(obj.model, 'saved_models/SVM5.pkl') 
clf = joblib.load('saved_models/SVM5.pkl') 
obj.predict_df(df_cross, clf)