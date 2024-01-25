# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 18:11:40 2022

@author: dguti
"""

from pycoral.utils import edgetpu
from pycoral.utils import dataset
from pycoral.adapters import common
from pycoral.adapters import classify
import numpy as np
from PIL import Image
from keras.preprocessing import image
from time import sleep
import time
from picamera import PiCamera

inicio=time.time()

camera = PiCamera()
camera.resolution = (320,320)
camera.rotation = 180
camera.start_preview(fullscreen=False, window=(30,30,320,240))
for i in range(1,4):
    print(4-i)
    sleep(1)
camera.capture('/home/pi/Clasificador-residuos/imagen.jpg')
camera.stop_preview()
camera.close()

# modelo, etiquetas e imagen de ejemplo
model_file = "modelxception_edgetpu.tflite"
path = "imagen.jpg"

# Crear un diccionario que mapea los índices de clase a nombres de clase
categories = {0: 'battery', 1: 'biological', 2: 'brown-glass', 3: 'cardboard', 4: 'clothes', 5: 'green-glass',
              6: 'metal', 7: 'paper', 8: 'plastic', 9: 'shoes', 10: 'trash',
              11: 'white-glass'}

# Inicializar el interprete de tensorflow lite
interpreter = edgetpu.make_interpreter(model_file)
# reserva de los tensores
interpreter.allocate_tensors()

# Ajustamos datos entrada
#size = common.input_size(interpreter) # datos entrada
#image = Image.open(image_file).convert('RGB').resize(size, Image.ANTIALIAS)

img = image.load_img(path, target_size=(320, 320))

# Conversión de numpy array
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
# Ejecutar inferencia
common.set_input(interpreter, x)
interpreter.invoke()
classes = classify.get_classes(interpreter, top_k=1)

fin=time.time()

diferencia=fin-inicio

print("Tiempo de ejecucion: ", diferencia, " segundos")

print("Resultado inferencia probabilidad: ", classes[0].score)

print("Resultado inferencia clase: ", categories[classes[0].id])


