from sklearn import datasets
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC

X, y = features, labels
clf = OneVsRestClassifier(LinearSVC(C=100.))
model = clf.fit(X, y)
output = model.predict(X)