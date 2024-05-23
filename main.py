import os

import cv2
import numpy as np
from matplotlib import pyplot as plt

param_1 = 20
param_2 = 31


def segmentar_iris(img):
    # Borra um pouco a imagem para auxiliar o processamento
    roi = cv2.medianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 5)

    threshold = cv2.threshold(roi, 100, 255, cv2.THRESH_BINARY)[1]

    circles = None
    min_radius = 80
    max_radius = 200

    while circles is None or circles.size == 0:
        circles = cv2.HoughCircles(threshold, cv2.HOUGH_GRADIENT,
                                   2,
                                   2000,
                                   param1=param_1,
                                   param2=param_2,
                                   minRadius=min_radius,
                                   maxRadius=max_radius)
        max_radius = max_radius + 10

    circles = np.uint16(np.around(circles))

    # Crie uma máscara com o mesmo tamanho da imagem
    mascara = np.zeros_like(img)

    # itera os circulos obtidos e desenha na imagem
    for i in circles[0, :]:
        # Preencha a área da iris na máscara com branco
        cv2.circle(mascara, (i[0], i[1]), i[2], (255, 255, 255), -1)

    # Aplique a máscara à imagem
    return cv2.bitwise_and(img, mascara)


def remover_pupila(img):
    # Borra um pouco a imagem para auxiliar o processamento
    roi = cv2.medianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 5)

    threshold = cv2.threshold(roi, 30, 255, cv2.THRESH_BINARY)[1]

    circles = cv2.HoughCircles(threshold,
                               cv2.HOUGH_GRADIENT,
                               2,
                               500,
                               param1=param_1,
                               param2=param_2,
                               minRadius=20,
                               maxRadius=50)

    circles = np.uint16(np.around(circles))

    # Crie uma máscara com o mesmo tamanho da imagem
    mascara = np.full_like(img, 255)

    # itera os circulos obtidos e desenha na imagem
    for i in circles[0, :]:
        # Preencha a área da pupila na máscara com preto
        cv2.circle(mascara, (i[0], i[1]), i[2], (0, 0, 0), -1)

    # Aplique a máscara à imagem
    return cv2.bitwise_and(img, mascara)


if __name__ == '__main__':
    for imgFile in os.listdir('result'):
        os.remove('result/' + imgFile)

    for fileName in os.listdir('data'):
        img = cv2.imread('data/' + fileName)
        iris = segmentar_iris(img)
        iris_sem_pupila = remover_pupila(iris)

        plt.imshow(iris_sem_pupila)
        plt.savefig('result/' + fileName, format='jpg')
