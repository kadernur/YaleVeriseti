# -*- coding: utf-8 -*-
"""
Created on Wed Feb  2 20:01:57 2022

@author: kader
"""

import pandas as pd
from sklearn.metrics import accuracy_score
from PIL import Image

#NumPy, Python programlama dili için bir kütüphanedir; büyük, çok boyutlu diziler ve matrisler için destek ekler ve bu 
#dizilerde çalışacak geniş bir üst düzey matematiksel işlev koleksiyonu sunar.
import numpy as np
import glob



def build_dataset():
	org_dataset = []
	labels = []
   
	for i in range(1, 16):
		filelist = glob.glob('./data/subject'+str(i).zfill(2)+"*")
		for fname in filelist:
			img = Image.open(fname)
            
			img = np.array(img.resize((32, 32), Image.ANTIALIAS))
			img = img.reshape(img.shape[0] * img.shape[1])
           
			org_dataset.append(img)
			labels.append(i)
	return np.array(org_dataset), np.array(labels)

data, labels = build_dataset()


data = data/255
print(len(data))


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size = 0.33, shuffle=True, random_state=42, stratify=labels)



from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=1)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print('Accuracy = '  + str(accuracy_score(y_test, y_pred)))



from sklearn.model_selection import train_test_split
import math


def dist(x1, x2):
    total = 0
    for i, j in zip(x1, x2):
        total += math.pow(i - j, 2)

    return math.sqrt(total)


def get_nearest_neighbors(row0, x_train, k):
    distances = []
    neighbors = []
    for i, row1 in enumerate(x_train):
        c = dist(row0, row1)
        distances.append([c, i])
        
    distances.sort(key = lambda x: x[0])
    for j in range(k):
          neighbors.append(distances[j])

    return neighbors



def KNN(K, X_test, X_train, y_train):
    
    Y_predict = []

    for x_test in X_test:
        neighbors = get_nearest_neighbors(x_test, X_train, K)
        targets = []
        for n in neighbors:
            index = n[1]
            targets.append(y_train[index])

        Y_predict.append(max(targets, key = targets.count))

    return Y_predict


y_pred = KNN(1, X_test, X_train, y_train)


from sklearn.metrics import r2_score


y_pred = KNN(1, X_test, X_train, y_train)
print(r2_score(y_pred, y_test))


import lpproj
lppModel = lpproj.LocalityPreservingProjection(n_components = 80, n_neighbors = 1)
selfObject = lppModel.fit(X_train)
trainKlpp = np.dot(X_train, selfObject.projection_)
testKlpp = np.dot(X_test, selfObject.projection_)

from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=1)
model.fit(trainKlpp, y_train)
y_pred = model.predict(testKlpp)
print('Accuracy with LPP = '  + str(accuracy_score(y_test, y_pred)))


import lle
vt = lle(X_train, 7, 80) 
X_trainn =  np.dot(X_train, np.transpose(vt))
X_testn =  np.dot(X_test, np.transpose(vt))
y_pred = KNN(1, X_testn, X_trainn, y_train)
from sklearn.metrics import r2_score
print(r2_score(y_test, y_pred))
print('Accuracy  With LLE = '  + str(accuracy_score(y_test, y_pred)))












