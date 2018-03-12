# -*- coding: utf-8 -*-
"""
Created on Sun Mar 11 17:16:21 2018

@author: Andy
"""

import pandas as pd

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()

counter = 0

for chunk in pd.read_csv('train.csv', chunksize=1000000):
    y_train = chunk['is_attributed']
    X_train = chunk.drop(['attributed_time','click_time','is_attributed'],axis=1)
    
    X_train = X_train.values
    y_train = y_train.values
    
    knn.fit(X_train,y_train)
    
    counter += 1
    print(f"Chunksize #{counter}:  {counter} million rows completed.")
    
    
    
test = pd.read_csv('train_sample.csv')
y_test = test['is_attributed']
X_test = test.drop(['attributed_time','click_time','is_attributed'],axis=1)
    
X_test = X_test.values
y_test = y_test.values

print(knn.score(X_test,y_test)) 