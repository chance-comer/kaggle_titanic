# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 15:53:14 2018

@author: kazantseva
"""

import pandas as pd
import numpy as np
import sklearn.model_selection as ms
import seaborn
import matplotlib.pyplot as plt

from sklearn import metrics, preprocessing
from sklearn import linear_model, svm, neighbors, ensemble
#import XGBoost as xgb

#---------------------------------------
# TEACH YOUR MODEL ON WHOLE SET OF TRAIN DATA
#---------------------------------------

data = pd.read_csv('train.csv')

y = data['Survived']
X = data.drop('Survived', axis = 1)

X = X.drop(['PassengerId', 'Name'], axis = 1)

X_train, X_test, y_train, y_test = ms.train_test_split(X, y, test_size = 0.3, random_state = 0)

category_features = ['Sex', 'Pclass', 'Cabin', 'Embarked', 'Ticket']
numeric_features = ['Age', 'SibSp', 'Parch', 'Fare']

X_train['Age'] = X_train['Age'].fillna(0)

scaler = preprocessing.StandardScaler()
X_train[numeric_features] = scaler.fit_transform(X_train[numeric_features])

X_train[category_features] = X_train[category_features].fillna('NoData')

to_plot_data = X_train[numeric_features]
to_plot_data['Survived'] = y_train

#seaborn.pairplot(to_plot_data, hue = 'Survived')

to_plot_data = X_train[category_features]
to_plot_data['Survived'] = y_train
#plt.figure(figsize = (5, 15))
#plt.subplot(5, 1, 1)
#seaborn.FacetGrid(to_plot_data['Sex'])
#plt.subplot(5, 1, 2)
#seaborn.FacetGrid(to_plot_data['Pclass'])
#plt.subplot(5, 1, 3)
#seaborn.FacetGrid(to_plot_data['Cabin'])
#plt.subplot(5, 1, 4)
#seaborn.FacetGrid(to_plot_data['Embarked'])
#plt.subplot(5, 1, 5)
#seaborn.FacetGrid(to_plot_data['Ticket'])
#plt.tight_layout()
#linear_model = lm.LogisticRegression()
seaborn.FacetGrid(to_plot_data, col='Sex')