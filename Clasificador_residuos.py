# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 12:38:12 2024

@author: vicen
"""

import numpy as np
import keras.applications.xception as xception
import tensorflow.keras as keras
import tensorflow as tf
from keras.layers import  Dense
from keras.models import Sequential
from keras.preprocessing import image
from keras.layers import Lambda
from picamera import PiCamera
from time import sleep

camera = PiCamera()
camera.resolution = (320,320)
camera.rotation = 180
camera.start_preview(fullscreen=False, window=(30,30,320,240))
for i in range(1,4):
    print(4-i)
    sleep(1)
camera.capture('/home/pi/imagen.jpg')
camera.stop_preview()
camera.close()


# Crear un diccionario que mapea los índices de clase a nombres de clase
categories = {0: 'battery', 1: 'biological', 2: 'brown-glass', 3: 'cardboard', 4: 'clothes', 5: 'green-glass',
              6: 'metal', 7: 'paper', 8: 'plastic', 9: 'shoes', 10: 'trash',
              11: 'white-glass'}

path='imagen.jpg'
IMAGE_WIDTH = 320
IMAGE_HEIGHT = 320
IMAGE_SIZE=(IMAGE_WIDTH, IMAGE_HEIGHT)
IMAGE_CHANNELS = 3

xception_layer = xception.Xception(include_top = False, input_shape = (IMAGE_WIDTH, IMAGE_HEIGHT,IMAGE_CHANNELS),
                       weights = 'imagenet')

# We don't want to train the imported weights
xception_layer.trainable = False


model = Sequential()
model.add(keras.Input(shape=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS)))

#create a custom layer to apply the preprocessing
def xception_preprocessing(img):
  return xception.preprocess_input(img)

model.add(Lambda(xception_preprocessing))

model.add(xception_layer)
model.add(tf.keras.layers.GlobalAveragePooling2D())
model.add(Dense(len(categories), activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])



# Restore the weights
#model.load_weights('modelxception.h5')

model.load_weights('modelxception.h5')

# Visualizar la arquitectura del modelo
model.summary()



img = image.load_img(path, target_size=(320, 320))

# Conversión de numpy array
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)

prediction = model.predict(x)

# Obtener el índice de la clase predicha
predicted_class_index = np.argmax(prediction)


# Obtener el nombre de la clase predicha
predicted_class_name = categories[predicted_class_index]

print("La clase predicha es:", predicted_class_name)

