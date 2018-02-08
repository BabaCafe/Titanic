import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('train.csv')

data = data.replace({'Embarked': {'S': 1, 'C': 2, 'Q': 3}, 'Sex': {'male': 1, 'female': 2}})
data.loc[data['Cabin'].notna(), 'Cabin'] = 1
data = data.fillna({'Cabin': 0})
data = data.fillna({'Age': data['Age'].mean()})

datatest = pd.read_csv('test.csv')

datatest = datatest.replace({'Embarked': {'S': 1, 'C': 2, 'Q': 3}, 'Sex': {'male': 1, 'female': 2}})
datatest.loc[datatest['Cabin'].notna(), 'Cabin'] = 1
datatest = datatest.fillna({'Cabin': 0})
datatest = datatest.fillna({'Age': data['Age'].mean()})
datatest = datatest.fillna(datatest.mean())
y = data['Survived'].ravel()

X = data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
X_test = datatest[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
X_test = (X_test - X.mean()) / X.std()
X = (X - X.mean()) / X.std()

X_test = pd.get_dummies(X_test, columns=['Sex', 'Pclass', 'Embarked'])
X_test.insert(0, 'Cabin', value=data['Cabin'])


X = pd.get_dummies(X, columns=['Sex', 'Pclass', 'Embarked'])
X.insert(0, 'Cabin', value=data['Cabin'])
#X = X.iloc[0:650, ]
#y = y[0:650]
from sklearn import svm

svc = svm.SVC(C=10, gamma=0.03)
svc.fit(X, y)
predict = svc.predict(X_test)
print(type(predict))
predict = np.array(predict)
print(X_test.shape, predict.shape)

Prediction = pd.DataFrame({'PassengerId': datatest['PassengerId'], 'Survived': predict})
print(Prediction.head())
Prediction.to_csv('Prediction650SVM_10_03.csv', index=False)
