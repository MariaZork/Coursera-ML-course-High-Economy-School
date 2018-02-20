# -*- coding: utf-8 -*-
"""
Created on Sun Feb 18 00:12:16 2018

author: Maria Zorkaltseva
"""
#====Задание по программированию: Выбор числа соседей====

#В этом задании вам нужно подобрать оптимальное значение k для алгоритма kNN. 
#Будем использовать набор данных Wine, где требуется предсказать сорт 
#винограда, из которого изготовлено вино, используя результаты 
#химических анализов.
#
#Выполните следующие шаги:
#    
#1. Загрузите выборку Wine по адресу 
#https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data 
#(файл также приложен к этому заданию)
#
#2. Извлеките из данных признаки и классы. Класс записан в первом столбце 
#(три варианта), признаки — в столбцах со второго по последний. 
#Более подробно о сути признаков можно прочитать по адресу 
#https://archive.ics.uci.edu/ml/datasets/Wine 
#(см. также файл wine.names, приложенный к заданию)
#
#3. Оценку качества необходимо провести методом кросс-валидации 
#по 5 блокам (5-fold). Создайте генератор разбиений, который перемешивает 
#выборку перед формированием блоков (shuffle=True). Для воспроизводимости 
#результата, создавайте генератор KFold с фиксированным параметром 
#random_state=42. В качестве меры качества используйте 
#долю верных ответов (accuracy).
#
#4. Найдите точность классификации на кросс-валидации для метода k ближайших 
#соседей (sklearn.neighbors.KNeighborsClassifier), при k от 1 до 50. 
#При каком k получилось оптимальное качество? Чему оно равно 
#(число в интервале от 0 до 1)? 
#Данные результаты и будут ответами на вопросы 1 и 2.
#
#5. Произведите масштабирование признаков с помощью функции 
#sklearn.preprocessing.scale. Снова найдите оптимальное k на кросс-валидации.
#
#6. Какое значение k получилось оптимальным после приведения признаков к одному 
#масштабу? Приведите ответы на вопросы 3 и 4. 
#Помогло ли масштабирование признаков?

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

data = pd.read_table('wine.data', sep=',', header=None)

X = data.iloc[:, 1:]
Y = data[0]

# create 5 splits for cross-validation
kf = KFold(n_splits=5, random_state=42, shuffle=True)
# varying n_neighbors from 1 to 50
n_neighbors = np.arange(1, 51)
tuned_parameters = [{'n_neighbors': n_neighbors}]
# create k_means classifier
clf = KNeighborsClassifier()
# GridCV over n_neighbors and splits
GridCV = GridSearchCV(clf, param_grid = tuned_parameters, cv=kf, scoring='f1_macro', refit=False)
GridCV.fit(X, Y)

scores = GridCV.cv_results_['mean_test_score']

score, temp = 0, GridCV.cv_results_['mean_test_score']
for i in range(len(temp)):
    if temp[i] > score:
        score = temp[i]
        Index = i
        
print("===BEFORE SCALING===")
print("Max value of mean_test_score %.2f for %d neighbor" % (score, Index+1))

plt.figure().set_size_inches(8, 6)
plt.plot(n_neighbors, scores, 'b*')
# ============================================================================
# scaling data in X (mean = 0 std = 1)
stdSc = StandardScaler()
X = stdSc.fit_transform(X)

# repeat calculation after scaling
GridCV.fit(X, Y)

scores = GridCV.cv_results_['mean_test_score']
plt.plot(n_neighbors, scores, 'b.')

score, temp = 0, GridCV.cv_results_['mean_test_score']
for i in range(len(temp)):
    if temp[i] > score:
        score = temp[i]
        Index = i
        
print("===AFTER SCALING===")
print("Max value of mean_test_score %.2f for %d neighbor" % (score, Index+1))