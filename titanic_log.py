import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

data = pd.read_csv('train.csv')
print(data.head())
data = data.replace({'Embarked': {'S': 1, 'C': 2, 'Q': 3}, 'Sex': {'male': 1, 'female': 2}})
data.loc[data['Cabin'].notna(), 'Cabin'] = 1
data = data.fillna({'Cabin': 0})
data = data.fillna({'Age': data['Age'].mean()})

#data[['Last_Name', 'First_Name']] = data['Name'].str.split(',', expand=True)
y = data['Survived']

#X = data[['Last_Name', 'Sex', 'Age', 'Pclass']]
X = data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
X = (X - X.mean()) / X.std()
X.insert(0, 'Parch_age_Sex', value=X['Parch'] * X['Age'])
X.insert(0, 'Sex_Parch', value=X['Sex'] * X['Parch'])
X.insert(0, 'Sex_age', value=X['Sex'] * X['Age'])
#X.insert(0, 'Parch_SibSp', value=X['Parch'] *  X['SibSp'])
#X.insert(0, 'Fare_Pclass', value=X['Fare'] * X['Pclass'])
X.insert(0, 'Sex_SibSp', value=X['Sex'] * X['SibSp'])
X.insert(0, 'Pclass_age', value=X['Pclass'] * X['Age'])
#X.insert(0, 'Fare_age', value=X['Fare'] * X['Age'])
#X.insert(0, 'Embarked_Fare', value=X['Embarked'] * X['Fare'])
X = (X - X.mean()) / X.std()
X = pd.get_dummies(X, columns=['Sex', 'Pclass', 'Embarked'])
X.insert(0, 'Cabin', value=data['Cabin'])

#print(X.iloc[0:10, :])

print(X.head())
print(X.shape, y.shape, type(X), type(y))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
#X_train, X_test, y_train, y_test = X.iloc[0:700, :], X.iloc[700:, :], y.iloc[0:700, ], y.iloc[700:, ]
#print(X_train.shape, X_test.shape, y_test.shape, y_train.shape)

learningRate = 10
epsilon = 0


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def Costreg(theta, X, y, learningRate):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)

    first = np.multiply(-y, (np.log(sigmoid(X * theta.T) + epsilon)))
    second = np.multiply((1 - y), (np.log(1 - sigmoid(X * theta.T) + epsilon)))
    reg = (learningRate / (2 * len(X))) * np.sum(np.power(theta[:, 1:theta.shape[1]], 2))

    return np.sum(first - second) / len(X) + reg


def gradient(theta, X, y, learningRate):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    # parameters=int(theta.ravel().shape[1])
    # grad=np.zeros(parameters)

    error = sigmoid(X * theta.T) - y
    grad = ((X.T * error) / len(X)).T + ((learningRate / len(X)) * theta)
    grad[0, 0] = np.sum(np.multiply(error, X[:, 0])) / len(X)

    return np.array(grad).ravel()


X_train.insert(0, 'ones', 1)
X_test.insert(0, 'ones', 1)
cols = list(X_train.columns.values)
X_train = np.array(X_train.values)
X_test = np.array(X_test.values)
theta = np.zeros(X_train.shape[1])
print(theta.shape)
y_train = np.array(y_train.values).reshape(-1, 1)
y_test = np.array(y_test.values).reshape(-1, 1)

#print(y.shape, X.shape)
#print(Costreg(theta, X, y, learningRate))
#grad = gradient(theta, X, y, learningRate)
#print(grad.shape, grad)

import scipy.optimize as opt
theta_min = opt.fmin_bfgs(f=Costreg, x0=theta, fprime=gradient, args=(X_train, y_train, learningRate))
print(theta_min)

theta_min = np.matrix(theta_min)


def predict_survival(X, theta):
    p = sigmoid(X * theta.T)
    return [1 if x > 0.5 else 0 for x in p]


predict = predict_survival(X_test, theta_min)

correct = [1 if (a == 0 and b == 0) or (a == 1 and b == 1) else 0 for a, b in zip(predict, y_test)]
accuracy = (sum(map(int, correct)) / len(correct))
print(accuracy)
print('accuracy = {0}%'.format(accuracy * 100))
