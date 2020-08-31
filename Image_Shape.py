# Clase creada por Juan Pablo Guerrero para la clase de Procesamiento de imágenes y visión
# Pontificia Universidad Javeriana
import cv2
import numpy as np


class ImageShape:

    def __init__(self, height, width):  # constructor entra la altura y el ancho deseado para la imagen
        self.width = width
        self.height = height
        self.shape = 0                                                  # forma de figura generada aleatoriamente
        self.approx_shape = 'None'                                      # deducción de la froma de la figura
        self.image = np.zeros([self.height, self.width, 3], np.uint8)   # fondo negro de la imagen
        self.center = (self.width // 2, self.height // 2)               # coordenadas del centro de la imagen

    def generate_shape(self):
        choose = np.random.randint(0, 4, 1)         # genera un numero aleatorio entre 0 a 4
                                                    # según el numero aleatorio se selecciona un tipo de figura
        if choose == 0:  # it's a triangle
            self.shape = 'triangle'
            side = min(self.width, self.height)//2  # calcula uno de los lado del triangulo equilatero
            triangle_height = np.sqrt(np.power(side, 2) - np.power(side / 2, 2)).astype(int)    # lado de la mitad del triangulo
            pa = np.array([self.center[0] - (side // 2), self.center[1] + (triangle_height // 3)])  # primer punto del triangulo
            pb = np.array([self.center[0] + (side // 2), self.center[1] + (triangle_height // 3)])  # segundo punto del triangulo
            pc = np.array([self.center[0], self.center[1] - (2 * triangle_height // 3)])            # tercer punto del triangulo
            triangle_pnt = np.array([pa.astype(int), pb.astype(int), pc.astype(int)])   # array con las cordenadas de los tres puntos
            cv2.drawContours(self.image, [triangle_pnt], 0, (255, 255, 0), -1)  # se dibuja el triangulo a partir de los tres puntos indexados

        elif choose == 1:   # it's a square
            self.shape = 'square'
            side = min(self.width, self.height) // 2  # calcula el lado del cuadrado
            pa = np.array([self.center[0] - side // 2, self.center[1] - side // 2])     # primer punto del cuadrado
            pb = np.array([self.center[0] + side // 2, self.center[1] + side // 2])     # punto opuesto al primer punto

            cv2.rectangle(self.image, tuple(pa), tuple(pb), (255, 255, 0), -1)  # se dibuja el cuadrado a partir de dos puntos opuestos
            square_rot = cv2.getRotationMatrix2D(self.center, 40, 1)    # calcula la rotacion de una matriz a partir del angulo dado
            self.image = cv2.warpAffine(self.image, square_rot, (self.width, self.height))  # se hace una transformacion afín entre la imagen y el cuadrado

        elif choose == 2:
            self.shape = 'rectangle'
            pa = np.array([self.center[0] - self.center[0] // 4, self.center[1] - self.center[1] // 4]) # primer punto del rectangulo
            pb = np.array([self.center[0] + self.center[0] // 4, self.center[1] + self.center[1] // 4]) # punto opuesto al primero
            cv2.rectangle(self.image, tuple(pa), tuple(pb), (255, 255, 0), -1)  # se dibuja el rectangulo a partir de dos puntos opuestos

        elif choose == 3:   # it's a circle
            self.shape = 'circle'
            radius = min(self.width, self.height) // 4  # radio del circulo
            cv2.circle(self.image, self.center, radius, (255, 255, 0), -1 ) # se dibuja el circulo a partir del radio y del centro del circulo

    def show_shape(self, time):
        cv2.imshow("Shape", self.image)     # muestra la imagen (fondo + figura aleatoria)
        cv2.waitKey(time)   # se muestra durante "time" milisegundos

    def get_shape(self):
        print('shape is a {}'.format(self.shape))
        return self.image, self.shape   # retorna la imagen y un string con el tipo de figura generada aleatoriamente

    def what_shape(self):   # deducción del tipo de figura

        image_gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY) # se pasa a escala de grises la imagen generada
        ret, im_bw = cv2.threshold(image_gray, 0, 255, cv2.THRESH_BINARY)   # se binariza la imagen
        cnt, hierarchy = cv2.findContours(im_bw, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)    #se hallan los contornos de cada figura

        contour_image = self.image.copy() # se realiza una copia de la imagen para no sobre-escribirla

        for nc in cnt:  # para cada contorno
            adj = 0.01*cv2.arcLength(nc, True)  # variable para ajustar la aproximacion por poligonos
            approx_polly = cv2.approxPolyDP(nc, adj, True) # se halla un poligono que aproxima la forma de la figura
            cv2.drawContours(contour_image, [approx_polly], 0, (0, 0, 255), 2) # se dibujan estos poligonos sobre la imagen
                                                                                # copiada para visualizarlos
            # a partir del numero de puntos del poligono se determina que tipo de figura es
            if len(approx_polly) == 3: # si son tres poligonos la figura es un triangulo
                self.approx_shape = 'triangle'

            elif len(approx_polly) == 4: # si son cuatro poligonos puede ser un cuadrado o un rectangulo
                x, y, w, h = cv2.boundingRect(approx_polly) # se calcula la altura y ancho del poligono de 4 lados
                asp_ratio = float(w)/h # se encuentra la relación de aspecto del poligono (ancho/alto)
                if 0.95 <= asp_ratio <= 1.05: # si la relacion de aspecto es cercana a uno, es un cuadrado
                    self.approx_shape = 'square'
                else: # en el resto de casos sera un rectangulo
                    self.approx_shape = 'rectangle'

            elif len(approx_polly) >= 10: # si tiene mas diez puntos el poligono lo se concidera un circulo
                self.approx_shape = 'circle'

            return contour_image, self.approx_shape  # retorna la imagen con el poligono dibujado y un string con la forma de la figura