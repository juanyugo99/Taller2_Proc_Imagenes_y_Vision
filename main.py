# Taller 2 de la clase de Procesamiento de imágenes y visión
# Realizado por Juan Pablo Guerrero
# Pontificia Universidad Javeriana

import cv2
import numpy as np
from Image_Shape import *


if __name__ == '__main__':

    height, width = input('write height and width (separate with space): ').split(' ') # se solicita la dimensión de la imagen

    image = ImageShape(int(height), int(width)) # se crea el objeto

    image.generate_shape() # se genera la figura aleatoria

    image.show_shape(5000) # muestra la figura durante 5 segundos

    approx_image, approx_shape = image.what_shape() # se deduce el tipo de figura aleatoria generada

    actual_image, actual_shape = image.get_shape() # se pide el tipo de figura que genero realmente

    if approx_shape == actual_shape: # se compara la deducción con la figura real para saber si es realizada correctamente
        print('correct deduction, figure is a {}'.format(approx_shape))
    else:
        print('deduction error')

    # descomentar si se desea ver el poligono aproximado

    # images = cv2.hconcat([actual_image, approx_image])
    # cv2.imshow('comparison', images)
    # cv2.waitKey(0)
    
    