# -*- coding: utf-8 -*-
"""
Created on Fri Apr  6 19:05:10 2018

@author: Maria
"""

#==Задание по программированию: Градиентный бустинг над решающими деревьями==
# В рамках данного задания мы рассмотрим датасет с конкурса
# Predicting a Biological Response.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss

# 1. Загрузите выборку из файла gbm-data.csv с помощью pandas и преобразуйте ее
# в массив numpy (параметр values у датафрейма). В первой колонке файла
# с данными записано, была или нет реакция. Все остальные колонки (d1 - d1776)
# содержат различные характеристики молекулы, такие как размер, форма и т.д.
# Разбейте выборку на обучающую и тестовую, используя функцию train_test_split
# с параметрами test_size = 0.8 и random_state = 241.

data = pd.read_csv('gbm-data.csv', sep=',').values
X = data[:, 1:]
Y = data[:, 0]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.8, \
                                      random_state=241)

# 2. Обучите GradientBoostingClassifier с параметрами n_estimators=250,
# verbose=True, random_state=241 и для каждого значения learning_rate из списка
# [1, 0.5, 0.3, 0.2, 0.1] проделайте следующее:
#
# - Используйте метод staged_decision_function для предсказания качества
# на обучающей и тестовой выборке на каждой итерации.
#
# - Преобразуйте полученное предсказание с помощью сигмоидной функции
# по формуле 1 / (1 + e^{−y_pred}), где y_pred — предсказанное значение.
#
# - Вычислите и постройте график значений log-loss (которую можно посчитать
# с помощью функции sklearn.metrics.log_loss) на обучающей и тестовой выборках,
# а также найдите минимальное значение метрики и номер итерации,
# на которой оно достигается.

def log_loss_estimation(clf, X , Y):
    result = []
    for y_pred in clf.staged_decision_function(X):
        result.append(log_loss(Y, [1 / (1 + np.exp(-y)) for y in y_pred]))
    return result


learning_rate = [1, 0.5, 0.3, 0.2, 0.1]
logloss_train, logloss_test = [], []
min_loss_results = {}

for lr in learning_rate:
    clf = GradientBoostingClassifier(n_estimators=250, verbose=True, \
                                     random_state=241, learning_rate=lr)

    clf.fit(X_train, Y_train)

    logloss_train = log_loss_estimation(clf, X_train, Y_train)
    logloss_test = log_loss_estimation(clf, X_test, Y_test)

    minimum_test = min(logloss_test)
    index_test = logloss_test.index(minimum_test)
    min_loss_results[lr] = (minimum_test, index_test)

    plt.figure()
    plt.plot(logloss_train, 'r', linewidth=2)
    plt.plot(logloss_test, 'g', linewidth=2)
    plt.legend(['train', 'test'])

# 3. Как можно охарактеризовать график качества на тестовой выборке, начиная
# с некоторой итерации: переобучение (overfitting) или недообучение
# (underfitting)? В ответе укажите одно из слов overfitting либо underfitting.
# Ответ: overfitting

# 4. Приведите минимальное значение log-loss на тестовой выборке и номер
# итерации, на котором оно достигается, при learning_rate = 0.2.

print("Minimal log_loss on learning rate 0.2 = %.2f and index = %d " % \
                      (min_loss_results[0.2][0], min_loss_results[0.2][1]))

# 5. На этих же данных обучите RandomForestClassifier с количеством деревьев,
# равным количеству итераций, на котором достигается наилучшее качество у
# градиентного бустинга из предыдущего пункта, c random_state=241 и остальными
# параметрами по умолчанию. Какое значение log-loss на тесте получается у этого
# случайного леса? (Не забывайте, что предсказания нужно получать с помощью
# функции predict_proba. В данном случае брать сигмоиду от
# оценки вероятности класса не нужно)
clf = GradientBoostingClassifier(n_estimators=min_loss_results[0.2][1], \
                                                             random_state=241)
clf.fit(X_train, Y_train)
Y_pred = clf.predict_proba(X_test)
error = log_loss(Y_test, Y_pred)
print("Error on best case = %.2f" % (error))