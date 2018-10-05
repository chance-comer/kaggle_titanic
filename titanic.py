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

from sklearn import metrics, preprocessing
from sklearn import linear_model, svm, neighbors, ensemble
#import XGBoost as xgb

#---------------------------------------
# TEACH YOUR MODEL ON WHOLE SET OF TRAIN DATA
#---------------------------------------

data = pd.read_csv('train.csv')

y = data['Survived']
X = data.drop('Survived', axis = 1)

X = X.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis = 1)

category_features = ['Sex', 'Pclass', 'Embarked']
numeric_features = ['Age']#, 'SibSp', 'Parch', 'Fare']

X[category_features] = X[category_features].fillna('NoData')
X.loc[X['Pclass'] == 1, 'Pclass'] = 'First'
X.loc[X['Pclass'] == 2, 'Pclass']  = 'Second'
X.loc[X['Pclass'] == 3, 'Pclass']  = 'Third'

X['Age'] = X['Age'].fillna(X['Age'].median(axis = 0)) # X_train['Age'].mean()

X_train, X_test, y_train, y_test = ms.train_test_split(X, y, test_size = 0.3, random_state = 0)

scaler = preprocessing.StandardScaler()
#X_train[numeric_features] = scaler.fit_transform(X_train[numeric_features])
#X_test[numeric_features] = scaler.fit_transform(X_test[numeric_features])

encoder = DV(sparse = False)
encoded_train_data = encoder.fit_transform(X_train[category_features].T.to_dict().values())
encoded_test_data = encoder.transform(X_test[category_features].T.to_dict().values())

#to_plot_data = X_train[numeric_features]
#to_plot_data['Survived'] = y_train

#seaborn.pairplot(to_plot_data, hue = 'Survived')

#to_plot_data = X_train[category_features]
#to_plot_data['Survived'] = y_train

#plt.figure(figsize = (5, 15))

#mosaic(to_plot_data, ['Sex', 'Pclass', 'Survived']);
#data.pivot_table('PassengerId', 'Pclass', 'Survived', 'count').plot(kind='bar', stacked=True)
#data.pivot_table('PassengerId', 'Sex', 'Survived', 'count').plot(kind='bar', stacked=True)
#data.pivot_table('PassengerId', 'Embarked', 'Survived', 'count').plot(kind='bar', stacked=True)
#encoder_lbl_Sex = preprocessing.LabelEncoder()
#encoded_lbl_Sex = encoder_lbl_Sex.fit_transform(X_train['Sex'])

#encoder_lbl_Pclass = preprocessing.LabelEncoder()
#encoded_lbl_Pclass = encoder_lbl_Sex.fit_transform(X_train['Pclass'])

#encoder_lbl_Embarked = preprocessing.LabelEncoder()
#encoded_lbl_Embarked = encoder_lbl_Sex.fit_transform(X_train['Embarked'])

#encoded_lbl_train = np.vstack((encoded_lbl_Sex, encoded_lbl_Pclass, encoded_lbl_Embarked))

#encoder_oh = preprocessing.OneHotEncoder(handle_unknown='ignore')
#encoded_X_train = encoder_oh.fit_transform(encoded_lbl_train.T).toarray()

whole_train_matrix = np.hstack((X_train[numeric_features], encoded_train_data))
whole_test_matrix = np.hstack((X_test[numeric_features], encoded_test_data))

linear_model_titanic = linear_model.LogisticRegression()
linear_model_titanic.fit(whole_train_matrix, y_train)
#predictions = linear_model_titanic.predict_proba(whole_test_matrix)

score_logistic = ms.cross_val_score(linear_model_titanic, whole_test_matrix, y_test, cv = 3).mean()

RF_model_titanic = ensemble.RandomForestClassifier(n_estimators = 100)
RF_model_titanic.fit(whole_train_matrix, y_train)

score_RF = ms.cross_val_score(RF_model_titanic, whole_test_matrix, y_test, cv = 3).mean()

svm_ = svm.OneClassSVM(gamma=10, nu=0.16) 
svm_.fit(whole_train_matrix)
labels = svm_.predict(whole_train_matrix)

seaborn.pairplot(pd.DataFrame(whole_train_matrix))