# -*- coding: utf-8 -*-
"""
Created on Fri Apr  6 17:55:27 2018

@author: Maria
"""

import pandas as pd
import numpy as np

from sklearn.model_selection import KFold, GridSearchCV
from sklearn.ensemble import RandomForestRegressor

#====Задание по программированию: Размер случайного леса====
# В этом задании вам нужно проследить за изменением качества случайного леса 
# в зависимости от количества деревьев в нем.

# 1. Загрузите данные из файла abalone.csv. Это датасет, в котором требуется 
# предсказать возраст ракушки (число колец) по физическим измерениям.

data = pd.read_csv('abalone.csv', sep=',')

# 2. Преобразуйте признак Sex в числовой: значение F должно перейти в -1, 
# I — в 0, M — в 1. Если вы используете Pandas, то подойдет следующий код: 

data['Sex'] = data['Sex'].map(lambda x: 1 if x == 'M' \
                                            else (-1 if x == 'F' else 0))

# 3. Разделите содержимое файлов на признаки и целевую переменную. В последнем
# столбце записана целевая переменная, в остальных — признаки.

X_train = data.iloc[:, :-1]
Y_train = data.iloc[:, -1]

# 4.Обучите случайный лес (sklearn.ensemble.RandomForestRegressor) с различным 
# числом деревьев: от 1 до 50 (не забудьте выставить "random_state=1" 
# в конструкторе). Для каждого из вариантов оцените качество работы полученного 
# леса на кросс-валидации по 5 блокам. Используйте параметры "random_state=1" и 
# "shuffle=True" при создании генератора кросс-валидации 
# sklearn.cross_validation.KFold. В качестве меры качества воспользуйтесь 
# коэффициентом детерминации (sklearn.metrics.r2_score).

kf = KFold(n_splits=5, random_state=1, shuffle=True)
n_estimators = np.arange(1, 51)
tuned_parameters = [{'n_estimators': n_estimators}]

Regressor = RandomForestRegressor(random_state=1)

GridCV = GridSearchCV(Regressor, param_grid = tuned_parameters, cv=kf, \
                      scoring='r2')
GridCV.fit(X_train, Y_train)

scores = GridCV.cv_results_['mean_test_score']

# 5. Определите, при каком минимальном количестве деревьев случайный лес 
# показывает качество на кросс-валидации выше 0.52. Это количество 
# и будет ответом на задание.

number = 1
while scores[number-1] <= 0.52:
    number += 1
    
print("Amount of trees =", number)
    
# 6. Обратите внимание на изменение качества по мере роста числа деревьев. 
# Ухудшается ли оно?
# Ответ: нет    