# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 15:53:14 2018

@author: kazantseva
"""

import pandas as pd
import numpy as np
import sklearn.model_selection as ms
import matplotlib.pyplot as plt
import seaborn
from  sklearn.feature_extraction import DictVectorizer as DV
from sklearn.grid_search import GridSearchCV
#import xgboost

from sklearn import metrics, preprocessing
from sklearn import linear_model, svm, neighbors, ensemble
#import XGBoost as xgb

#---------------------------------------
# TEACH YOUR MODEL ON WHOLE SET OF TRAIN DATA
#---------------------------------------

data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

y = data['Survived']
X = data.drop('Survived', axis = 1)

X = X.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis = 1)
X_test = test_data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis = 1)

category_features = ['Sex']#, 'Pclass']#, 'Embarked']
numeric_features = ['Age', 'Pclass']#, 'SibSp', 'Parch', 'Fare']

X[category_features] = X[category_features].fillna('NoData')
#X.loc[X['Pclass'] == 1, 'Pclass'] = 'First'
#X.loc[X['Pclass'] == 2, 'Pclass']  = 'Second'
#X.loc[X['Pclass'] == 3, 'Pclass']  = 'Third'

X_test[category_features] = X_test[category_features].fillna('NoData')
#X_test.loc[X_test['Pclass'] == 1, 'Pclass'] = 'First'
#X_test.loc[X_test['Pclass'] == 2, 'Pclass']  = 'Second'
#X_test.loc[X_test['Pclass'] == 3, 'Pclass']  = 'Third'

X['Age'] = X['Age'].fillna(X['Age'].median(axis = 0)) # X_train['Age'].mean()
X_test['Age'] = X_test['Age'].fillna(X['Age'].median(axis = 0)) 

#X_train, X_test, y_train, y_test = ms.train_test_split(X, y, test_size = 0.3, random_state = 0, shuffle = False)
X_train = X

scaler = preprocessing.StandardScaler()
X_train.loc[:, numeric_features] = scaler.fit_transform(X_train[numeric_features])
X_test.loc[:, numeric_features] = scaler.transform(X_test[numeric_features])

encoder = DV(sparse = False)
encoded_train_data = encoder.fit_transform(X_train[category_features].T.to_dict().values())
encoded_test_data = encoder.transform(X_test[category_features].T.to_dict().values())

whole_train_matrix = np.hstack((X_train[numeric_features], encoded_train_data))
whole_test_matrix = np.hstack((X_test[numeric_features], encoded_test_data))

param_log = {
        'penalty': ['l2'],
        'C': [0.01, 0.1, 1, 10],
        'solver' : [ 'liblinear', 'newton-cg', 'lbfgs', 'sag', 'saga'],
        'max_iter': [100, 200]
        }

linear_model_titanic = linear_model.LogisticRegression(random_state = 0)
#linear_search_res = ms.GridSearchCV(linear_model_titanic, param_log)
linear_model_titanic.fit(whole_train_matrix, y)
#linear_search_res.fit(whole_train_matrix, y_train)
predictions = linear_model_titanic.predict_proba(whole_test_matrix)
predictions = [1 if prediction[1] > 0.62 else 0 for prediction in predictions]
#score_logistic = ms.cross_val_score(linear_model_titanic, whole_test_matrix, y_test, cv = 3).mean()
#score_search_logistic = ms.cross_val_score(linear_search_res, whole_test_matrix, y_test, cv = 3).mean()

#RF_model_titanic = ensemble.RandomForestClassifier(n_estimators = 100)
#RF_model_titanic.fit(whole_train_matrix, y)

#score_RF = ms.cross_val_score(RF_model_titanic, whole_test_matrix, y_test, cv = 3).mean()
#predictions = RF_model_titanic.predict(whole_test_matrix)
#KN_model_titanic = neighbors.KNeighborsClassifier(n_neighbors = 5)
#KN_model_titanic.fit(whole_train_matrix, y)
#score_KN = ms.cross_val_score(KN_model_titanic, whole_test_matrix, y_test, cv = 3).mean()
#predictions = linear_model_titanic.predict(whole_test_matrix)

answer = pd.DataFrame()
answer['PassengerId'] = test_data['PassengerId']
answer['Survived'] = predictions
answer.to_csv('a.csv', index = False)
#Radius_model_titanic = neighbors.RadiusNeighborsClassifier(radius = 1, outlier_label=1)
#Radius_model_titanic.fit(whole_train_matrix, y_train)
#score_radius = ms.cross_val_score(Radius_model_titanic, whole_test_matrix, y_test, cv = 3).mean()
#'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'
param_SVM = {'kernel': ['rbf', 'sigmoid'],
        'C': [1],
        'degree': [1, 2, 3],
        'coef0': [0, 0.01, 0.1, 1, 5],
        'shrinking': [True, False]}

#SVM_model_titanic = svm.SVC(random_state = 0)
#search_param_res = ms.GridSearchCV(SVM_model_titanic, param_SVM, cv = 3)
#SVM_model_titanic.fit(whole_train_matrix, y_train)
#search_param_res.fit(whole_train_matrix, y_train)
#score_SVM = ms.cross_val_score(SVM_model_titanic, whole_test_matrix, y_test, scoring = 'accuracy', cv = 3).mean()
#score_SVM_gridSearch = ms.cross_val_score(search_param_res, whole_test_matrix, y_test, scoring = 'accuracy', cv = 3).mean()

#xgb_model_titanic = xgboost.XGBClassifier()
#xgb_model_titanic.fit(whole_train_matrix, y_train)
#score_xgb = ms.cross_val_score(xgb_model_titanic, whole_test_matrix, y_test, cv = 3).mean()
#svm_ = svm.OneClassSVM(gamma=10, nu=0.01) 
#svm_.fit(np.array(X['Age']).reshape(-1,1))
#labels = svm_.predict(np.array(X['Age']).reshape(-1,1))


