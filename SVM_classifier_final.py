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
import random

le = LabelEncoder()
df_data = pd.read_hdf("data_extract/features_dataset_2b.h5")
clist = ["chroma{}".format(i) for i in range(12)]
lclist = ["lc{}".format(i) for i in range(12)]
df_data[lclist] = np.log(df_data[clist])
flist = ["mfcc{}".format(i) for i in range(14)]+lclist+["centroid","crest","flatness","kurtosis","mean"]

plot = dict()
num_iter = 5
kernel = 'poly'

for k in range(num_iter):
	for num_per in [5,10,15,20,25,30,35]:
		ids = [i for i in range(40)]
		for i in [5,31,35,39]:
		    ids.remove(i)
		random.shuffle(ids)
		sel = [ids[i] for i in range(len(ids)) if i<num_per]
		bool_arr = np.array([(row in sel) for row in df_data['id']])
		df = df_data[bool_arr]
		for train_per in [0.02,0.05,0.1,0.2,0.3,0.4]:
			df_train, df_cross, df_test = get_partitions_2(df,train_per=train_per,cross_per=(0.9-train_per),test_per=0.1)
			obj = SVM(flist,kernel=kernel)
			obj.fit(df_train)
			#obj = MultiGauss(flist)
			#obj.fit(df_train)
			output = obj.predict_df(df_test)
			x = float(filter(None, output.split('\n')[-2].split(' '))[-2])
			print num_per,train_per,x
			if k==0:
				plot[(num_per,train_per)] = (x/float(num_iter))
			else:
				plot[(num_per,train_per)] += (x/float(num_iter))

print plot
joblib.dump(plot, 'outputs/SVM_plot_poly.pkl')
#obj2 = joblib.load('outputs/MGauss_plot.pkl')
#print obj2