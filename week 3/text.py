# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 00:35:28 2018

author: Maria Zorkaltseva
"""

#====Задание по программированию: Анализ текстов====
#1. Загрузите объекты из новостного датасета 20 newsgroups, относящиеся к 
#категориям "космос" и "атеизм" (инструкция приведена выше). Обратите внимание, 
#что загрузка данных может занять несколько минут
#
#2. Вычислите TF-IDF-признаки для всех текстов. Обратите внимание, что в этом 
#задании мы предлагаем вам вычислить TF-IDF по всем данным. При таком подходе 
#получается, что признаки на обучающем множестве используют информацию из 
#тестовой выборки — но такая ситуация вполне законна, поскольку мы не используем
#значения целевой переменной из теста. На практике нередко встречаются ситуации, 
#когда признаки объектов тестовой выборки известны на момент обучения, и поэтому 
#можно ими пользоваться при обучении алгоритма.
#
#3. Подберите минимальный лучший параметр C из множества 
#[10^-5, 10^-4, ... 10^4, 10^5] для SVM с линейным ядром (kernel='linear') 
#при помощи кросс-валидации по 5 блокам. Укажите параметр random_state=241 
#и для SVM, и для KFold. В качестве меры качества 
#используйте долю верных ответов (accuracy).
#
#4. Обучите SVM по всей выборке с оптимальным параметром C, 
#найденным на предыдущем шаге.
#
#5. Найдите 10 слов с наибольшим абсолютным значением веса (веса хранятся в 
#поле coef_ у svm.SVC). Они являются ответом на это задание. Укажите эти слова 
#через запятую или пробел, в нижнем регистре, в лексикографическом порядке.

import numpy as np

from sklearn import datasets
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold, GridSearchCV

newsgroups = datasets.fetch_20newsgroups(
                    subset='all', 
                    categories=['alt.atheism', 'sci.space']
             )

X = newsgroups.data
Y = newsgroups.target

# TF-IDF calculation at all data
textSc = TfidfVectorizer()
X = textSc.fit_transform(X)

grid = {'C': np.power(10.0, np.arange(-5, 6))}
cv = KFold(n_splits=5, shuffle=True, random_state=241)
clf = SVC(kernel='linear', random_state=241)
gs = GridSearchCV(clf, grid, scoring='accuracy', cv=cv)
gs.fit(X, Y)

# find optimal C
max_score = 0
for a in gs.grid_scores_:
    if a.mean_validation_score > max_score:
        max_score = a.mean_validation_score
        opt_C = a.parameters['C']
print(opt_C)

# learning with optimal C
clf = SVC(kernel='linear', random_state=241, C=opt_C)
clf.fit(X, Y)

# 10 words with max absolute weights
feature_mapping = textSc.get_feature_names()

temp_dict = dict()
for i in range(len(feature_mapping)):
    word = feature_mapping[i]
    weight = clf.coef_[0, i]
    temp_dict.update({word: abs(weight)})

temp_dict = sorted(temp_dict.items(), key=lambda x: -x[1])

ten_words_dict = dict()
for i in range(10):
    ten_words_dict.update({temp_dict[i][0]: temp_dict[i][1]})
    
ten_words_dict = sorted(ten_words_dict.items(), key=lambda x: x[0])

for i in range(len(ten_words_dict)):
    print(ten_words_dict[i][0], end=' ')