import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

data = pd.read_csv('train.csv')
print(data.head())
#data = data.replace({'Embarked': {'S': 1, 'C': 2, 'Q': 3}, 'Sex': {'male': 1, 'female': 2}})
print(data.head())
data = data.fillna({'Age': data['Age'].mean()})
data = data.replace({'Embarked': {'S': 1, 'C': 2, 'Q': 3}, 'Sex': {'male': 1, 'female': 2}})
data.loc[data['Cabin'].notna(), 'Cabin'] = 1
data = data.fillna({'Cabin': 0})
X = data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
y = data['Survived'].ravel()
X = (X - X.mean()) / X.std()
X = pd.get_dummies(X, columns=['Sex', 'Pclass', 'Embarked'])
X.insert(0, 'Cabin', value=data['Cabin'])
print(X.head())
#X = (X - X.mean()) / X.std()
# print(X.head())
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
#X_train, X_test, y_train, y_test = X.iloc[0:700, :], X.iloc[700:, :], y[:700], y[700:]

from sklearn import svm

best_score = 0
best_params = {'C': None, 'gamma': None}

C_values = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 60, 100]
gamma_values = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 60, 100]

for C in C_values:
    for gamma in gamma_values:
        svc = svm.SVC(C=C, gamma=gamma)
        svc.fit(X_train, y_train)
        score = svc.score(X_test, y_test)

        if score > best_score:
            best_score = score
            best_params['C'] = C
            best_params['gamma'] = gamma

print(best_score, best_params)

svc = svm.SVC(C=10, gamma=0.03)
svc.fit(X_train, y_train)
print(svc.score(X_test, y_test))
