# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 16:20:03 2018

@author: Maria
"""

#====Тренировочное задание по программированию: Уменьшение количества цветов изображения====
import numpy as np

from skimage.io import imread, imshow, show
from skimage import img_as_float
from sklearn.cluster import KMeans

# 1. Загрузите картинку parrots.jpg. Преобразуйте изображение, приведя все
# значения в интервал от 0 до 1. Для этого можно воспользоваться функцией
# img_as_float из модуля skimage. Обратите внимание на этот шаг, так как при
# работе с исходным изображением вы получите некорректный результат.

image = imread('parrots.jpg')
image = img_as_float(image)

# 2. Создайте матрицу объекты-признаки: характеризуйте каждый пиксель тремя
# координатами - значениями интенсивности в пространстве RGB.

X = image.reshape((image.shape[0]*image.shape[1], image.shape[2]))

# 3. Запустите алгоритм K-Means с параметрами init='k-means++' и random_state=241.
# После выделения кластеров все пиксели, отнесенные в один кластер, попробуйте
# заполнить двумя способами: медианным и средним цветом по кластеру.

cls = KMeans(init='k-means++', random_state=241)
cls.fit(X)
labels = cls.labels_

def median_color(X_original, labels):
    num_clusters = labels.max() + 1
    X_median_color = np.zeros(X_original.shape)
    for label in range(num_clusters):
        indices = np.where(labels==label)
        median_color_index = len(X_original[indices]) // 2
        X_median_color[indices] = X_original[median_color_index]  
    return X_median_color
    
def average_color(X_original, labels):
    num_clusters = labels.max() + 1
    X_average_color = np.zeros(X_original.shape)
    for label in range(num_clusters):
        indices = np.where(labels==label)
        average_color = sum(X_original[indices]) / len(X_original[indices])
        X_average_color[indices] = average_color
    return X_average_color


X_median_color = median_color(X, labels)
X_average_color = average_color(X, labels)


original_image = imshow(image)
show()
im_median_color = imshow(X_median_color.reshape(image.shape[0], image.shape[1], image.shape[2]))
show()
im_average_color = imshow(X_average_color.reshape(image.shape[0], image.shape[1], image.shape[2]))
show()

# 4. Измерьте качество получившейся сегментации с помощью метрики PSNR. Эту метрику
# нужно реализовать самостоятельно.

def PSNR_metric(im_original, im_noisy):
    MSE = 1/len(im_original)*sum((im_original - im_noisy)**2)
    PSNR = 20*np.log10(im_original.max()) - 10*np.log10(MSE)
    return PSNR

PSNR_median_color = PSNR_metric(X, X_median_color)
PSNR_average_color = PSNR_metric(X, X_average_color)
print("PSNR (in RGB space) when median color was used:", PSNR_median_color)
print("PSNR (in RGB space) when average color was used:", PSNR_average_color)

# 5. Найдите минимальное количество кластеров, при котором значение PSNR выше 20
# (можно рассмотреть не более 20 кластеров, но не забудьте рассмотреть оба
# способа заполнения пикселей одного кластера). Это число и будет
# ответом в данной задаче.

n_clusters = np.arange(1, 21)
for n_cluster in n_clusters:
    cls = KMeans(init='k-means++', random_state=241, n_clusters=n_cluster)
    cls.fit(X)
    labels = cls.labels_
    X_median_color = median_color(X, labels)
    X_average_color = average_color(X, labels)
    PSNR_median_color = PSNR_metric(X, X_median_color)
    PSNR_average_color = PSNR_metric(X, X_average_color)
    if np.any(PSNR_median_color > 20) or np.any(PSNR_average_color > 20):
        print("Minimal number of clusters:", n_cluster+1)
        break

    






