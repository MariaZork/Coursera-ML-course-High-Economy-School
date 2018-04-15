# -*- coding: utf-8 -*-
"""
Created on Wed Apr  4 19:54:23 2018

@author: Maria
"""
#Задание по программированию: Линейная регрессия: прогноз оклада по описанию вакансии

#1. Загрузите данные об описаниях вакансий и соответствующих годовых зарплатах 
#из файла salary-train.csv (либо его заархивированную версию salary-train.zip).

#2. Проведите предобработку:
#Приведите тексты к нижнему регистру (text.lower()).
#Замените все, кроме букв и цифр, на пробелы — это облегчит дальнейшее 
#разделение текста на слова. Для такой замены в строке text подходит следующий 
#вызов: re.sub('[^a-zA-Z0-9]', ' ', text). Также можно воспользоваться методом 
#replace у DataFrame, чтобы сразу преобразовать все тексты:

#Примените TfidfVectorizer для преобразования текстов в векторы признаков. 
#Оставьте только те слова, которые встречаются хотя бы в 5 объектах 
#(параметр min_df у TfidfVectorizer).

#Замените пропуски в столбцах LocationNormalized и ContractTime на специальную 
#строку 'nan'. Код для этого был приведен выше.

#Примените DictVectorizer для получения one-hot-кодирования признаков 
#LocationNormalized и ContractTime.

#Объедините все полученные признаки в одну матрицу "объекты-признаки". 
#Обратите внимание, что матрицы для текстов и категориальных признаков являются 
#разреженными. Для объединения их столбцов нужно воспользоваться 
#функцией scipy.sparse.hstack.

#3. Обучите гребневую регрессию с параметрами alpha=1 и random_state=241. 
#Целевая переменная записана в столбце SalaryNormalized.

#4. Постройте прогнозы для двух примеров из файла salary-test-mini.csv. 
#Значения полученных прогнозов являются ответом на задание. 
#Укажите их через пробел.

import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import Ridge
from scipy.sparse import hstack

data = pd.read_csv('salary-train.csv', sep=',')
test_data = pd.read_csv('salary-test-mini.csv', sep=',')

# texts in first column to low reqister
data['FullDescription'] = data['FullDescription'].str.lower()
test_data['FullDescription'] = test_data['FullDescription'].str.lower()

# replace all symbols (not letters and numbers) on spaces
data['FullDescription'] = \
            data['FullDescription'].replace('[^a-zA-Z0-9]', ' ', regex = True)
test_data['FullDescription'] = \
            test_data['FullDescription'].replace('[^a-zA-Z0-9]', ' ', regex = True)
            
# Apply TfidfVectorizer to convert texts into feature vectors. 
# Leave only those words that occur in at least 5 objects 
# (min_df parameter of TfidfVectorizer).
textSc = TfidfVectorizer(min_df=5)
X_train_textSc = textSc.fit_transform(data['FullDescription'])
X_test_textSc = textSc.transform(test_data['FullDescription'])

# Replace the omissions in the columns LocationNormalized and ContractTime 
# with a special string 'nan'  
data['LocationNormalized'].fillna('nan', inplace=True)
data['ContractTime'].fillna('nan', inplace=True)

# binarize categorical features
enc = DictVectorizer()
X_train_categ = enc.fit_transform(data[['LocationNormalized', 'ContractTime']].to_dict('records'))
X_test_categ = enc.transform(test_data[['LocationNormalized', 'ContractTime']].to_dict('records'))

X_train_union = hstack([X_train_textSc, X_train_categ])
X_test_union = hstack([X_test_textSc, X_test_categ])

regressor = Ridge(alpha=1, random_state=241)
regressor.fit(X_train_union, data['SalaryNormalized'])

Y_pred = regressor.predict(X_test_union)
print("Predicted salaries: %.2f %.2f" % (Y_pred[0], Y_pred[1]))