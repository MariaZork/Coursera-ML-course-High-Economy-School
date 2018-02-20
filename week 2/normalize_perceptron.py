# -*- coding: utf-8 -*-
"""
Created on Sun Feb 18 21:35:16 2018

author: Maria Zorkaltseva
"""

#====Задание по программированию: Нормализация признаков====

#1. Загрузите обучающую и тестовую выборки из файлов perceptron-train.csv и
#perceptron-test.csv. Целевая переменная записана в первом столбце, признаки —
#во втором и третьем.
#
#2. Обучите персептрон со стандартными параметрами и random_state=241.
#
#3. Подсчитайте качество (долю правильно классифицированных объектов, accuracy)
#полученного классификатора на тестовой выборке.
#
#4. Нормализуйте обучающую и тестовую выборку с помощью класса StandardScaler.
#
#5. Обучите персептрон на новой выборке. Найдите долю правильных ответов на
#тестовой выборке.
#
#6. Найдите разность между качеством на тестовой выборке после нормализации и
#качеством до нее. Это число и будет ответом на задание.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

train_data = pd.read_csv('perceptron-train.csv', header=None)
test_data = pd.read_csv('perceptron-test.csv', header=None)
X_train = train_data.iloc[:, 1:]
Y_train = train_data[0]
X_test = test_data.iloc[:, 1:]
Y_test = test_data[0]

clf = Perceptron(random_state=241)
clf.fit(X_train, Y_train)
Y_preds = clf.predict(X_test)
score_0 = accuracy_score(Y_test, Y_preds)
print("===BEFORE SCALING===")
print("Accuracy (num of Y_true/num of Y_preds positivies):", score_0)

# scaling data in X (mean = 0 std = 1)
stdSc = StandardScaler()
X_train = stdSc.fit_transform(X_train)
X_test = stdSc.transform(X_test)
clf.fit(X_train, Y_train)
Y_preds = clf.predict(X_test)
score_1 = accuracy_score(Y_test, Y_preds)
print("===AFTER SCALING===")
print("Accuracy (num of Y_true/num of Y_preds positivies):", score_1)
print("Difference beetween before and after scaling scores:", abs(score_1 - score_0))
