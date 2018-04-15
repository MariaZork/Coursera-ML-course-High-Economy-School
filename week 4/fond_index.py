# -*- coding: utf-8 -*-
"""
Created on Thu Apr  5 16:58:57 2018

@author: Maria
"""

import pandas as pd
import numpy as np

from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV

# 1. Загрузите данные close_prices.csv. В этом файле приведены цены акций 30 
# компаний на закрытии торгов за каждый день периода.

data = pd.read_csv('close_prices.csv', sep=',')
X = data.iloc[:, 1:]
#2. На загруженных данных обучите преобразование PCA с числом компоненты 
#равным 10. Скольких компонент хватит, чтобы объяснить 90% дисперсии?

summa, n_components = 0, 1  
while summa < 0.9:
    decomposition = PCA(n_components=n_components)
    decomposition.fit(X)
    var = decomposition.explained_variance_ratio_
    summa = sum(var)
    n_components += 1
    
print("Amount of components for 90 % variance:", n_components - 1)

#3. Примените построенное преобразование к исходным данным и возьмите 
#значения первой компоненты.
X = decomposition.transform(X)
X1 = X[:, 0]

#3. Загрузите информацию об индексе Доу-Джонса из файла djia_index.csv. 
#Чему равна корреляция Пирсона между первой компонентой и индексом Доу-Джонса?
data_djia = pd.read_csv('djia_index.csv', sep=',')
corr_coef = np.corrcoef(X1, data_djia.iloc[:, 1:].values.ravel())
print("Correlation coefficient: %.2f" % (corr_coef[1][0]))

#5. Какая компания имеет наибольший вес в первой компоненте? Укажите ее 
#название с большой буквы.
comp_0 = pd.Series(decomposition.components_[0])
max_index = comp_0.sort_values(ascending=False).index[0]
print("Name of company with max weight in component 1: %s" % (data.columns[max_index+1]))
