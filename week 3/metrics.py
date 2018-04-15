# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 23:55:27 2018

@author: Maria
"""
#====Задание по программированию: Метрики качества классификации====
#1. Загрузите файл classification.csv. В нем записаны истинные классы объектов 
#выборки (колонка true) и ответы некоторого классификатора (колонка pred).
#2. Заполните таблицу ошибок классификации:
#    Actual Positive	Actual Negative
#    Predicted Positive	TP	FP
#    Predicted Negative	FN	TN
#Для этого подсчитайте величины TP, FP, FN и TN согласно их определениям. 
#Например, FP — это количество объектов, имеющих класс 0, но отнесенных 
#алгоритмом к классу 1. Ответ в данном вопросе — четыре числа через пробел.
#
#3. Посчитайте основные метрики качества классификатора:
#
#Accuracy (доля верно угаданных) — sklearn.metrics.accuracy_score
#Precision (точность) — sklearn.metrics.precision_score
#Recall (полнота) — sklearn.metrics.recall_score
#F-мера — sklearn.metrics.f1_score
#В качестве ответа укажите эти четыре числа через пробел.
#
#4. Имеется четыре обученных классификатора. В файле scores.csv записаны 
#истинные классы и значения степени принадлежности положительному классу для 
#каждого классификатора на некоторой выборке:
#
#для логистической регрессии — вероятность положительного 
#класса (колонка score_logreg),
#для SVM — отступ от разделяющей поверхности (колонка score_svm),
#для метрического алгоритма — взвешенная сумма классов 
#соседей (колонка score_knn),
#для решающего дерева — доля положительных объектов 
#в листе (колонка score_tree). Загрузите этот файл.
#
#5. Посчитайте площадь под ROC-кривой для каждого классификатора. 
#Какой классификатор имеет наибольшее значение метрики 
#AUC-ROC (укажите название столбца)? Воспользуйтесь функцией 
#sklearn.metrics.roc_auc_score.
#
#6. Какой классификатор достигает наибольшей точности (Precision) 
#при полноте (Recall) не менее 70% ?
#
#Чтобы получить ответ на этот вопрос, найдите все точки precision-recall-кривой
#с помощью функции sklearn.metrics.precision_recall_curve. Она возвращает 
#три массива: precision, recall, thresholds. В них записаны точность и полнота 
#при определенных порогах, указанных в массиве thresholds. 
#Найдите максимальной значение точности среди тех записей, для которых полнота 
#не меньше, чем 0.7.
 
import pandas as pd
import numpy as np

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, \
recall_score, f1_score, roc_auc_score, precision_recall_curve

data = pd.read_csv('classification.csv', sep=',')

tn, fp, fn, tp = confusion_matrix(data['true'], data['pred']).ravel()
accuracy = accuracy_score(data['true'], data['pred'])
precision = precision_score(data['true'], data['pred'])
recall = recall_score(data['true'], data['pred'])
f1 = f1_score(data['true'], data['pred'])

print("tn: %d fp: %d fn: %d tp: %d" % (tn, fp, fn, tp))
print("accuracy: %.2f\nprecision: %.2f\nrecall: %.2f\nf1_score: %.2f\n" \
                                          % (accuracy, precision, recall, f1))

data = pd.read_csv('scores.csv', sep=',')
    
auc_logreg = roc_auc_score(data['true'], data['score_logreg'])
auc_svm = roc_auc_score(data['true'], data['score_svm'])
auc_knn = roc_auc_score(data['true'], data['score_knn'])
auc_tree = roc_auc_score(data['true'], data['score_tree'])
                   
print("auc_log_regression: %.2f\nauc_svm: %.2f\nauc_knn: %.2f\nauc_tree: %.2f" \
      % (auc_logreg, auc_svm, auc_knn, auc_tree))

auc_scores = sorted({'score_logreg': auc_logreg, 'score_svm': auc_svm, \
    'score_knn': auc_knn, 'score_tree': auc_tree}.items(), key=lambda x: -x[1])

print("Maximum of auc score for column name: %s\n" % (auc_scores[0][0]))

def max_precision(precision, recall):
    maximum = 0
    for i in range(len(recall)):
        if recall[i] >= 0.7 and precision[i] > maximum:
            maximum = precision[i]           
        else:
            continue
    return maximum
        
precision, recall, thresholds = \
                    precision_recall_curve(data['true'], data['score_logreg'])
max_precision_logreg = max_precision(precision, recall)
precision, recall, thresholds = \
                    precision_recall_curve(data['true'], data['score_svm'])
max_precision_svm = max_precision(precision, recall)
precision, recall, thresholds = \
                    precision_recall_curve(data['true'], data['score_knn'])
max_precision_knn = max_precision(precision, recall)
precision, recall, thresholds = \
                    precision_recall_curve(data['true'], data['score_tree'])
max_precision_tree = max_precision(precision, recall)

print("max_precision_logreg: %.2f\nmax_precision_svm: %.2f\nmax_precision_knn: %.2f\nmax_precision_tree: %.2f" \
      % (max_precision_logreg, max_precision_svm, max_precision_knn, max_precision_tree))
                
max_precision = sorted({'score_logreg': max_precision_logreg, 'score_svm': max_precision_svm, \
    'score_knn': max_precision_knn, 'score_tree': max_precision_tree}.items(), key=lambda x: -x[1])

print("Maximum precision when recall >=0.7 for column name: %s\n" % (max_precision[0][0]))