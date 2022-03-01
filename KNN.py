# -*- coding: utf-8 -*-
"""
Created on Tue Dec 14 14:03:46 2021

@author: kader
"""

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, confusion_matrix
import math

class KNN:
    def __init__(self, k_value):
        self.k = k_value

    def dist(self, row0, row1): # euclid
        total = 0
        for i, j in zip(row0, row1):
            total += math.pow(i-j, 2)
        
        return math.sqrt(total)

    def get_nearest_neighbors(self, row_to_search):
        
        distances, neighbors = [], [] 
        for i, x_row in enumerate(self.x_train):
            d = self.dist(row_to_search, x_row)
            distances.append([d, i]) # dist, index
        
        distances.sort(key = lambda x: x[0])
        
        for i in range(self.k):
            neighbors.append(distances[i])

        return neighbors

    def predict(self, X_test, X_train, Y_train):


        self.x_train, self.y_train = X_train, Y_train

        y_predict = []

        for x_row in X_test:

            neighbors = self.get_nearest_neighbors(x_row)
            targets = []
            for n in neighbors:
                ind = n[1]
                targets.append(self.y_train[ind])

            y_predict.append(max(targets, key = targets.count))

        return y_predict
   
    
    