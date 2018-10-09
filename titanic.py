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
import re
#import xgboost

#"Mr","Mrs","Miss","Master","Don","Rev","Dr","Mme","Ms","Major","Lady","Sir",
#"Mlle","Col","Capt","Countess","Jonkheer"

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

X = X.drop(['PassengerId'], axis = 1)
X_test = test_data.drop(['PassengerId'], axis = 1)

X['Family'] = X['SibSp'] + X['Parch']
X_test['Family'] = X_test['SibSp'] + X_test['Parch']
X['isCabin'] = X['Cabin'].map(lambda x: 'y' if x != x else 'n')
X_test['isCabin'] = X_test['Cabin'].map(lambda x: 'y' if x != x else 'y')
X['Title'] = [re.match('[\s\S]*(Mrs|Mr|Miss|Master|Don|Rev|Dr|Mme|Ms|Major|Lady|Sir|Mlle|Col|Capt|Countess|Jonkheer)', name).group(1) for name in X['Name']]
X_test['Title'] = [re.match('[\s\S]*(Mrs|Mr|Miss|Master|Don|Rev|Dr|Mme|Ms|Major|Lady|Sir|Mlle|Col|Capt|Countess|Jonkheer)', name).group(1) for name in X_test['Name']]
X['Ticket_prefix'] = [re.search('[\w\W]+ ', ticket).group(0).strip() if re.search('[\w\W]+ ', ticket) else 'NoPrefix' for ticket in X['Ticket']]
X['Ticket_number'] = [re.search(' [\d]+|[\d]+', ticket).group(0).strip() if re.search(' [\d]+|[\d]+', ticket) else 0 for ticket in X['Ticket']]
X['Ticket_number'] = X['Ticket_number'].astype('int')
X['Odd'] = [t%2 for t in X['Ticket_number']]

X_test['Ticket_prefix'] = [re.search('[\w\W]+ ', ticket).group(0).strip() if re.search('[\w\W]+ ', ticket) else 'NoPrefix' for ticket in X_test['Ticket']]
X_test['Ticket_number'] = [re.search(' [\d]+', ticket).group(0).strip() if re.search(' [\d]+', ticket) else 0 for ticket in X_test['Ticket']]
X_test['Ticket_number'] = X_test['Ticket_number'].astype('int')

X.loc[X['Ticket_prefix'].isin(["A./5.", "A.5.", "A/5", "A/5.", "A/S",\
      "A/4", "A/4.", "A4.", 'AQ/4', 'A. 2.', 'AQ/3.', 'S.C./A.4.', 'SC/A4', 'Fa']), 'Ticket_prefix'] = 'A4A5'
#X.loc[X['Ticket_prefix'].isin(["A/4", "A/4.", "A4."]), 'Ticket_prefix'] = 'A4'
X.loc[X['Ticket_prefix'].isin(["C.A.", "CA", "CA.", "W./C.", "W/C", 'C.A./SOTON']), 'Ticket_prefix'] = 'CA'
X.loc[X['Ticket_prefix'].isin(["W.E.P."]), 'Ticket_prefix'] = 'WE/P'
X.loc[X['Ticket_prefix'].isin(["SW/PP", "S.W./PP", 'P/PP', 'PP', 'LP']), 'Ticket_prefix'] = 'P/SW/PP'
X.loc[X['Ticket_prefix'].isin(["STON/O 2.", "STON/O2."]), 'Ticket_prefix'] = 'SOTON/O2'
X.loc[X['Ticket_prefix'].isin(["SOTON/O.Q.", 'STON/OQ.']), 'Ticket_prefix'] = 'SOTON/OQ'
X.loc[X['Ticket_prefix'].isin(["S.O.C.", 'S.O./P.P.', 'S.O.P.', 'S.P.']), 'Ticket_prefix'] = 'SO/C'
X.loc[X['Ticket_prefix'].isin(["SC/AH Basle", 'SC/PARIS', 'S.C./PARIS',\
           'SC/Paris', 'SC/AH', 'SC', 'SCO/W', 'C']), 'Ticket_prefix'] = 'SC'

X_test.loc[X_test['Ticket_prefix'].isin(["A./5.", "A.5.", "A/5", "A/5.", "A/S",\
           "A/4", "A/4.", "A4.", 'AQ/4', 'A. 2.', 'AQ/3.', 'S.C./A.4.', 'SC/A4', 'Fa']), 'Ticket_prefix'] = 'A4A5'
X_test.loc[X_test['Ticket_prefix'].isin(["A/4", "A/4.", "A4."]), 'Ticket_prefix'] = 'A4'
X_test.loc[X_test['Ticket_prefix'].isin(["C.A.", "CA", "CA.", "W./C.", "W/C", 'C.A./SOTON']), 'Ticket_prefix'] = 'CA'
X_test.loc[X_test['Ticket_prefix'].isin(["W.E.P."]), 'Ticket_prefix'] = 'WE/P'
X_test.loc[X_test['Ticket_prefix'].isin(["SW/PP", "S.W./PP", 'P/PP', 'PP', 'LP']), 'Ticket_prefix'] = 'P/SW/PP'
X_test.loc[X_test['Ticket_prefix'].isin(["STON/O 2.", "STON/O2."]), 'Ticket_prefix'] = 'SOTON/O2'
X_test.loc[X_test['Ticket_prefix'].isin(["SOTON/O.Q.", 'STON/OQ.']), 'Ticket_prefix'] = 'SOTON/OQ'
X_test.loc[X_test['Ticket_prefix'].isin(["S.O.C.", 'S.O./P.P.', 'S.O.P.']), 'Ticket_prefix'] = 'SO/C'
X_test.loc[X_test['Ticket_prefix'].isin(["SC/AH Basle", 'SC/PARIS', 'S.C./PARIS',\
           'SC/Paris', 'SC/AH', 'SC', 'SCO/W', 'SC/A.3', 'C']), 'Ticket_prefix'] = 'SC'
#  A/4, A/4., A4.,
# A./5., A.5., A/5, A/5., A/S,
# C, 
# C.A., CA, CA.,
# C.A./SOTON,  F.C., F.C.C., Fa, 
# P/PP, LINE, PC, PP, S.C./A.4.,  
# S.O./P.P.,  S.O.P., S.P., SC, SCO/W,
# SC/AH, SC/AH Basle, 
# SO/C,  S.O.C.,
# S.C./PARIS, SC/PARIS, SC/Paris
# S.W./PP, SW/PP,
# W./C., W/C, 
# SOTON/O.Q., SOTON/OQ,
# SOTON/O2,  STON/O 2., STON/O2., 
# W.E.P., WE/P
prefix_count = pd.value_counts(X['Ticket_prefix'])
survived_prefix = [{ prefix: [
  {'total' : prefix_count[prefix]},
  {'survived' : [list(pd.value_counts(y[X[X['Ticket_prefix'] == prefix].index]).index), list(pd.value_counts(y[X[X['Ticket_prefix'] == prefix].index]).values)] },
  {'class': [list(pd.value_counts(data.iloc[X[X['Ticket_prefix'] == prefix].index]['Pclass']).index), list(pd.value_counts(data.iloc[X[X['Ticket_prefix'] == prefix].index]['Pclass']).values)]}]
  }  for prefix in prefix_count.index] 

prefix_count_test = pd.value_counts(X_test['Ticket_prefix'])
survived_prefix_test = [{ prefix: [
  {'total' : prefix_count_test[prefix]},
  #{'survived' : [list(pd.value_counts(y[X[X['Ticket_prefix'] == prefix].index]).index), list(pd.value_counts(y[X[X['Ticket_prefix'] == prefix].index]).values)] },
  {'class': [list(pd.value_counts(test_data.iloc[X_test[X_test['Ticket_prefix'] == prefix].index]['Pclass']).index), list(pd.value_counts(test_data.iloc[X_test[X_test['Ticket_prefix'] == prefix].index]['Pclass']).values)]}]
  }  for prefix in prefix_count_test.index] 

X['Deck'] = [re.search('(\w)(\d{0,3}?)$', cabin).group(1) if cabin == cabin else 'NoData' for cabin in X['Cabin']]
X['Cabin_num'] = [re.search('(\w)(\d{0,3}?)$', cabin).group(2) if cabin == cabin and re.search('(\w)(\d{0,3}?)$', cabin).group(2) != '' else 0 for cabin in X['Cabin']]
X['Cabin_num'] = X['Cabin_num'].astype('int')

X = X.drop(['Name'], axis = 1)
X_test = X_test.drop(['Name'], axis = 1)

X.loc[X['Title'].isin(["Capt", "Col", "Don", "Dr", "Jonkheer", "Lady", "Major", "Rev", "Sir", "Countess"]), 'Title'] = 'Aristoctratic'
X.loc[X['Title'].isin(["Ms", "Mlle"]), 'Title'] = 'Miss'
X.loc[X['Title'].isin(["Mme", "Mlle"]), 'Title'] = 'Mrs'

X_test.loc[X_test['Title'].isin(["Capt", "Col", "Don", "Dr", "Jonkheer", "Lady", "Major", "Rev", "Sir", "Countess"]), 'Title'] = 'Aristoctratic'
X_test.loc[X_test['Title'].isin(["Ms", "Mlle"]), 'Title'] = 'Miss'
X_test.loc[X_test['Title'].isin(["Mme", "Mlle"]), 'Title'] = 'Mrs'

titles = ['Mr', 'Mrs', 'Miss', 'Master', 'Aristoctratic']

for title in titles:
  val_to_fill = X.loc[X['Title'] == title, 'Age'].median()
  X.loc[X['Title'] == title, 'Age'] =  X.loc[X['Title'] == title, 'Age'].fillna(val_to_fill)
  X_test.loc[X_test['Title'] == title, 'Age'] =  X_test.loc[X_test['Title'] == title, 'Age'].fillna(val_to_fill)

category_features = ['Sex', 'Title']#, 'Embarked']
numeric_features = ['Age', 'Pclass', 'Family', 'Fare']#, 'SibSp', 'Parch', 'Fare']

X[category_features] = X[category_features].fillna('NoData')
#X.loc[X['Pclass'] == 1, 'Pclass'] = 'First'
#X.loc[X['Pclass'] == 2, 'Pclass']  = 'Second'
#X.loc[X['Pclass'] == 3, 'Pclass']  = 'Third'

X_test[category_features] = X_test[category_features].fillna('NoData')
X_test['Fare'] = X_test['Fare'].fillna(X.loc[X['Pclass'] == 3, 'Fare'].median())
#X_test.loc[X_test['Pclass'] == 1, 'Pclass'] = 'First'
#X_test.loc[X_test['Pclass'] == 2, 'Pclass']  = 'Second'
#X_test.loc[X_test['Pclass'] == 3, 'Pclass']  = 'Third'

#X['Age'] = X['Age'].fillna(X['Age'].median(axis = 0)) # X_train['Age'].mean()
#X_test['Age'] = X_test['Age'].fillna(X['Age'].median(axis = 0)) 

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
        'penalty': ['l1', 'l2'],
        'C': [0.01, 0.1, 1, 2, 3, 4, 5, 10],
        'solver' : [ 'liblinear'],
        'max_iter': [100, 500],
        'tol': [0.0001, 0.00001, 0.00001],
        'class_weight': [None, 'balanced']
        }
#survived = y[y == 1]
#isCabin_surv = data.iloc[survived.index]['Cabin'].map(lambda x: 0 if x != x else 1)
#n_survived = y[y == 0]
#isCabin_n_surv = data.iloc[n_survived.index]['Cabin'].map(lambda x: 0 if x != x else 1)
#colors = ['blue','green']
#plt.hist([isCabin_surv, isCabin_n_surv],  histtype='bar', color=colors, stacked=True, fill=True, label = ['survived', 'not survived'])
#plt.legend()

linear_model_titanic = linear_model.LogisticRegression(random_state = 0)
linear_search_res = ms.GridSearchCV(linear_model_titanic, param_log)
linear_model_titanic.fit(whole_train_matrix, y)
linear_search_res.fit(whole_train_matrix, y)
predictions = linear_search_res.predict_proba(whole_test_matrix)
predictions = [1 if prediction[1] > 0.6 else 0 for prediction in predictions]
score_logistic = ms.cross_val_score(linear_model_titanic, whole_train_matrix, y, cv = 3).mean()
score_search_logistic = ms.cross_val_score(linear_search_res.best_estimator_, whole_train_matrix, y, cv = 3).mean()
answer = pd.DataFrame()
answer['PassengerId'] = test_data['PassengerId']
answer['Survived'] = predictions
answer.to_csv('a.csv', index = False)

RF_model_titanic = ensemble.RandomForestClassifier(n_estimators = 100)
#RF_model_titanic.fit(whole_train_matrix, y)

score_RF = ms.cross_val_score(RF_model_titanic, whole_train_matrix, y, cv = 3).mean()
#predictions = RF_model_titanic.predict(whole_test_matrix)
KN_model_titanic = neighbors.KNeighborsClassifier(n_neighbors = 5)
#KN_model_titanic.fit(whole_train_matrix, y)
score_KN = ms.cross_val_score(KN_model_titanic, whole_train_matrix, y, cv = 3).mean()
#predictions = KN_model_titanic.predict_proba(whole_test_matrix)
#predictions = [1 if prediction[1] > 0.6 else 0 for prediction in predictions]
#answer = pd.DataFrame()
#answer['PassengerId'] = test_data['PassengerId']
#answer['Survived'] = predictions
#answer.to_csv('a.csv', index = False)
#Radius_model_titanic = neighbors.RadiusNeighborsClassifier(radius = 1, outlier_label=1)
#Radius_model_titanic.fit(whole_train_matrix, y_train)
#score_radius = ms.cross_val_score(Radius_model_titanic, whole_test_matrix, y_test, cv = 3).mean()
#'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'
param_SVM = {'kernel': ['rbf', 'sigmoid'],
        'C': [1],
        'degree': [1, 2, 3],
        'coef0': [0, 0.01, 0.1, 1, 5],
        'shrinking': [True, False]}

SVM_model_titanic = svm.SVC(random_state = 0, probability=True)
#search_param_res = ms.GridSearchCV(SVM_model_titanic, param_SVM, cv = 3)
SVM_model_titanic.fit(whole_train_matrix, y)
#search_param_res.fit(whole_train_matrix, y)
#score_SVM = ms.cross_val_score(SVM_model_titanic, whole_test_matrix, y_test, scoring = 'accuracy', cv = 3).mean()
#score_SVM_gridSearch = ms.cross_val_score(search_param_res.best_estimator_, whole_train_matrix, y, scoring = 'accuracy', cv = 3).mean()
score_SVM = ms.cross_val_score(SVM_model_titanic, whole_train_matrix, y, scoring = 'accuracy', cv = 3).mean()

print(str.format('logistic {0}, logisticGridCV {4} RF {1}, KN {2}, SVM {3}', \
                 score_logistic, score_RF, score_KN, score_SVM, \
                 score_search_logistic))
predictions = SVM_model_titanic.predict_proba(whole_test_matrix)
predictions = [1 if prediction[1] > 0.6 else 0 for prediction in predictions]
answer = pd.DataFrame()
answer['PassengerId'] = test_data['PassengerId']
answer['Survived'] = predictions
answer.to_csv('a.csv', index = False)
#xgb_model_titanic = xgboost.XGBClassifier()
#xgb_model_titanic.fit(whole_train_matrix, y_train)
#score_xgb = ms.cross_val_score(xgb_model_titanic, whole_test_matrix, y_test, cv = 3).mean()
#svm_ = svm.OneClassSVM(gamma=10, nu=0.01) 
#svm_.fit(np.array(X['Age']).reshape(-1,1))
#labels = svm_.predict(np.array(X['Age']).reshape(-1,1))


