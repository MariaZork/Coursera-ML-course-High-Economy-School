# -*- coding: utf-8 -*-
"""
Created on Sun Feb 18 18:39:45 2018

author: Maria Zorkaltseva
"""
#====Задание по программированию: Выбор метрики====
#
#Мы будем использовать в данном задании набор данных Boston, где нужно 
#предсказать стоимость жилья на основе различных характеристик расположения 
#(загрязненность воздуха, близость к дорогам и т.д.). Подробнее о признаках 
#можно почитать по адресу 
#https://archive.ics.uci.edu/ml/machine-learning-databases/housing/
#
#1. Загрузите выборку Boston с помощью функции sklearn.datasets.load_boston(). 
#Результатом вызова данной функции является объект, у которого признаки 
#записаны в поле data, а целевой вектор — в поле target.
#
#2. Приведите признаки в выборке к одному масштабу при помощи функции 
#sklearn.preprocessing.scale.
#
#3. Переберите разные варианты параметра метрики p по сетке от 1 до 10 с 
#таким шагом, чтобы всего было протестировано 200 вариантов 
#(используйте функцию numpy.linspace). Используйте KNeighborsRegressor 
#с n_neighbors=5 и weights='distance' — данный параметр добавляет в алгоритм 
#веса, зависящие от расстояния до ближайших соседей. В качестве метрики 
#качества используйте среднеквадратичную ошибку 
#(параметр scoring='mean_squared_error' у cross_val_score; при использовании 
#библиотеки scikit-learn версии 0.18.1 и выше необходимо указывать 
#scoring='neg_mean_squared_error'). Качество оценивайте, как и в предыдущем 
#задании, с помощью кросс-валидации по 5 блокам с random_state = 42, не 
#забудьте включить перемешивание выборки (shuffle=True).
#
#4. Определите, при каком p качество на кросс-валидации оказалось оптимальным. 
#Обратите внимание, что cross_val_score возвращает массив показателей качества 
#по блокам; необходимо максимизировать среднее этих показателей. 
#Это значение параметра и будет ответом на задачу.

# Ответ: при p = 1 

import numpy as np

from sklearn.model_selection import KFold, GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_boston

boston = load_boston()

# scaling data in X (mean = 0 std = 1)
stdSc = StandardScaler()
X = stdSc.fit_transform(boston.data)
# target vector
Y = boston.target

kf = KFold(n_splits=5, random_state=42, shuffle=True)
p = np.linspace(1, 10, 200)
tuned_parameters = [{'p': p}]

Regressor = KNeighborsRegressor(n_neighbors=5, weights='distance')

GridCV = GridSearchCV(Regressor, param_grid = tuned_parameters, cv=kf, \
                      scoring='neg_mean_squared_error', refit=False)
GridCV.fit(X, Y)

scores = GridCV.cv_results_