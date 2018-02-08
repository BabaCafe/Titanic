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

X = data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
y = data['Survived'].ravel()
X = (X - X.mean()) / X.std()
X = pd.get_dummies(X, columns=['Sex', 'Pclass', 'Embarked'])
print(X.head())
#X = (X - X.mean()) / X.std()
# print(X.head())
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
X_train, X_test, y_train, y_test = X.iloc[0:600, :], X.iloc[600:, :], y[0:600], y[600:]

from sklearn.neural_network import MLPClassifier
clf = MLPClassifier(solver='lbfgs', hidden_layer_sizes=(3,), activation='logistic')
clf.fit(X_train, y_train)

print(clf.score(X_test, y_test))
