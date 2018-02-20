# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 02:21:27 2018

author: Maria Zorkaltseva
"""

#====Задание по программированию: Предобработка данных в Pandas====

#Загрузите датасет titanic.csv и, используя описанные выше способы работы с
#данными, найдите ответы на вопросы

#1. Какое количество мужчин и женщин ехало на корабле? В качестве ответа
#приведите два числа через пробел.
#
#2. Какой части пассажиров удалось выжить? Посчитайте долю выживших пассажиров.
#Ответ приведите в процентах
#(число в интервале от 0 до 100, знак процента не нужен),
#округлив до двух знаков.
#
#3. Какую долю пассажиры первого класса составляли среди всех пассажиров?
#Ответ приведите в процентах
#(число в интервале от 0 до 100, знак процента не нужен),
#округлив до двух знаков.
#
#4. Какого возраста были пассажиры? Посчитайте среднее и медиану возраста
#пассажиров. В качестве ответа приведите два числа через пробел.
#
#5. Коррелируют ли число братьев/сестер/супругов с числом родителей/детей?
#Посчитайте корреляцию Пирсона между признаками SibSp и Parch.
#
#6. Какое самое популярное женское имя на корабле? Извлеките из полного
#имени пассажира (колонка Name) его личное имя (First Name). Это задание —
#типичный пример того, с чем сталкивается специалист по анализу данных.
#Данные очень разнородные и шумные, но из них требуется извлечь необходимую
#информацию. Попробуйте вручную разобрать несколько значений столбца Name и
#выработать правило для извлечения имен, а также разделения их
#на женские и мужские.

import pandas
data = pandas.DataFrame.from_csv('_titanic.csv')

##-----1-----
Count = data['Sex'].value_counts()
print (Count)
#
#-----2-----
CountSurvived = len(data[data['Survived'] == 1])
CountOfPassengers = len(data[data['Survived'] != 1])
print ('Value of Survived, in percents = %.2f' %
           (CountSurvived/(CountSurvived+CountOfPassengers)*100))
#
##-----3-----
PClass1 = len(data[data['Pclass'] == 1])
PClass = len(data)
print ("Value of 1 class passengers, in percents = %.2f" % (PClass1/PClass*100))
#
#
##-----4-----
Mean = data['Age'].mean()
Median = data['Age'].median()
print("Mean = %.2f Median = %.2f" % (Mean, Median))
##-----5-----
data.corr('pearson')
coeff = data['SibSp'].corr(data['Parch'])
print ("Correlation coeficient = %.2f"% coeff)

#-----6-----
data = data[data['Sex'] == 'female']
data = [x.split('.')[1].strip().lstrip('(').rstrip(')') for x in data["Name"]]
print(*data, sep='\n')

data = [x.split('(') for x in data]
print(*data, sep='\n')

temp = []
for x in data:
    if len(x) == 2:
        temp.append(x[1].split(' ')[0])
    else:
        temp.append(x[0].split(' ')[0])

S = pandas.Series(data=temp)
print(S.describe())


