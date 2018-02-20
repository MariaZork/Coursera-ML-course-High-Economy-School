# -*- coding: utf-8 -*-
"""
Created on Thu Jan 15 12:54:21 2018

author: Maria Zorkaltseva
"""

#====Задание по программированию: Важность признаков====
#
#1. Загрузите выборку из файла titanic.csv с помощью пакета Pandas.
#
#2. Оставьте в выборке четыре признака: класс пассажира (Pclass), 
#цену билета (Fare), возраст пассажира (Age) и его пол (Sex).
#
#3. Обратите внимание, что признак Sex имеет строковые значения.
#
#4. Выделите целевую переменную — она записана в столбце Survived.
#
#5. В данных есть пропущенные значения — например, для некоторых пассажиров 
#неизвестен их возраст. Такие записи при чтении их в pandas принимают значение 
#nan. Найдите все объекты, у которых есть пропущенные признаки, 
#и удалите их из выборки.
#
#6. Обучите решающее дерево с параметром random_state=241 и остальными 
#параметрами по умолчанию 
#(речь идет о параметрах конструктора DecisionTreeСlassifier).
#
#7. Вычислите важности признаков и найдите два признака с наибольшей важностью. 
#Их названия будут ответами для данной задачи (в качестве ответа укажите 
#названия признаков через запятую или пробел, порядок не важен).

import pandas as pd
from sklearn.tree import DecisionTreeClassifier

data = pd.DataFrame.from_csv('_titanic.csv')

# delete null values from age column
data = data[data['Age'].notnull()]
# processing of 'Sex' data: string to int
data['Sex'][lambda x: x == 'male'] = 0
data['Sex'][lambda x: x == 'female'] = 1

# prepare features X and target values Y from data
X = data.loc[:,['Pclass', 'Fare', 'Age', 'Sex']]
Y = data.loc[:,['Survived']]

clf = DecisionTreeClassifier(random_state=241)
clf.fit(X, Y)

importances = clf.feature_importances_