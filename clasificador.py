# -*- coding: utf-8 -*-
"""
@author: celllarod, vicguzher
"""

import numpy as np
import pandas as pd
import random
import os
import matplotlib.pyplot as plt
import seaborn as sns
import keras.applications.xception as xception
import zipfile
import sys
import time
import tensorflow.keras as keras
import tensorflow as tf
from PIL import Image
from keras.layers import Input, Conv2D, Dense, Flatten, MaxPooling2D, Input, GlobalAveragePooling2D
from keras.layers.experimental.preprocessing import Normalization
from keras.models import Model, Sequential
from keras.preprocessing import image
from keras.utils import to_categorical
from keras.layers import Lambda
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.applications.imagenet_utils import preprocess_input
import cv2
import random
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam

################################################################################
# Inicialización de los generadores de números aleatorios para reproducibilidad
# Valor de la semilla inicial
seed_value= 1

# 1. Definir la variable de entorno `PYTHONHASHSEED` a un valor dado
import os
os.environ['PYTHONHASHSEED']=str(seed_value)

# 2. Iniciar el generador de número aleatorios en `python` 
# módulo random
import random
random.seed(seed_value)

# 3. Iniciar el generador de número aleatorios de `numpy`
import numpy as np
np.random.seed(seed_value)

# 4. Iniciar el generador de número aleatorios de `tensorflow`
import tensorflow as tf
tf.random.set_seed(seed_value) #tf.random.set.seed(seed_value)
################################################################################

#%% 
path = "dataset/garbage_classification/"

# Diccionario con cada una de las clases y una etiqueta numérica
categories = {0: 'paper', 1: 'cardboard', 2: 'plastic', 3: 'metal', 4: 'trash', 5: 'battery',
              6: 'shoes', 7: 'clothes', 8: 'green-glass', 9: 'brown-glass', 10: 'white-glass',
              11: 'biological'}

#%% Creación del dataframe {rutaMuestra, categoria(nº)}
import re


def add_class_name_prefix(df, col_name):
   
    return df

# Lista para guardar todos los nombres de los ficheros del dataset
filenames_list = []
# Lista para guardar la categoría de cada fichero del dataset (valor numérico)
categories_list = []

# Cada carpeta del dataset contiene las muestras de una categoría
for category in categories:
    filenames = os.listdir(path + categories[category])

    filenames_list = filenames_list  + filenames
    categories_list = categories_list + [category] * len(filenames)

df = pd.DataFrame({
    'fichero': filenames_list,
    'categoria': categories_list
})

# Añadimos a cada fichero un prefijo con su categoría -->"paper104.jpg" se convierte en "paper/paper104.jpg"
df['fichero'] = df['fichero'].apply(lambda x: x[:re.search("\d",x).start()] + '/' + x)

# Barajamos el dataset aleatoriamente
# frac=1: conservar todas las filas
# reset_index: reestablece indices del df
# drop=True: evita que se añada fila al dataset con los índices antiguos
# TODO: pensar si esto se puede hacer directamente al hacer separacion train/test
# df = df.sample(frac=1).reset_index(drop=True)
# TODO: one-hot-encoding a labels


print('Número de muestras del dataset = ' , len(df))

#%%
df.head()

#%%
df.info()

#%% Ver imagen aleatoria del dataset
random_row = random.randint(0, len(df)-1)
sample = df.iloc[random_row]
randomimage = tf.keras.utils.load_img(path +sample['fichero'])
plt.title(sample['fichero'])
plt.imshow(randomimage)

#%% Visualizar el balanceo de clases
def visualizar_balanceo_clases(df, color):
    df_visualization = df.copy()
    # Convertir label numérica a string
    df_visualization['categoria'] = df_visualization['categoria'].apply(lambda x:categories[x] )
    df_visualization['categoria'].value_counts().plot.bar(x = 'count', y = 'categoria', color=color)
    
    plt.xlabel("Clases de residuos", labelpad=14)
    plt.ylabel("Nº imágenes", labelpad=14)
    plt.title("Nº imágenes por clase", y=1.02);
    
plt.figure(figsize=(12, 5))
visualizar_balanceo_clases(df, color='blue')

#%% Separar dataset en train (80%) y test (10%)

# TODO: no tiene mucho sentido hacer esto. etiquetas ya numericas
#Change the categories from numbers to names
# df["categoria"] = df["categoria"].replace(categories)

df_train, df_test = train_test_split(df, test_size=0.2, shuffle=True, stratify=df['categoria'], random_state=42)
# df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)  

df_train = df_train.reset_index(drop=True) 
df_test = df_test.reset_index(drop=True)

X_train = df_train['fichero']
X_test = df_test['fichero']
Y_train = df_train['categoria']
Y_test = df_test['categoria']

print('Train size = ',  df_train.shape[0] , '\nTest size = ',  df_test.shape[0])

plt.figure(figsize=(12, 5))
visualizar_balanceo_clases(df_train, color='blue')
visualizar_balanceo_clases(df_test, color='green')

X_train = df_train['fichero']
X_train = df_train['fichero']
#%% Generadores de datos
# ImageDataGenerator: pre-procesamiento y data augmentation (manipulación de imágenes en tiempo de entrenamiento
IMAGE_GENERATOR  = False
# if (IMAGE_GENERATOR):
    
    
#%% OPCION 1: Red neuronal convolucional 
# TODO: CN-Hands-Signs
X_train = X_train / 255.
X_test = X_test / 255.


model = Sequential()
model.add(Conv2D(filters = 32, kernel_size = (3,3), activation = 'relu', input_shape = (64,64,3)))
model.add(Conv2D(filters = 32, kernel_size = (3,3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = 2))
model.add(Conv2D(filters = 64, kernel_size = (3,3), activation = 'relu'))
model.add(Conv2D(filters = 64, kernel_size = (3,3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = 2))
model.add(Conv2D(filters = 128, kernel_size = (3,3), activation = 'relu'))
model.add(Conv2D(filters = 128, kernel_size = (3,3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = 2))
model.add(Flatten())
model.add(Dropout(rate = 0.5))
model.add(Dense(512, activation = 'relu'))
model.add(Dense(units = 6, activation = 'softmax'))

model.summary()


model.compile(optimizer = Adam(learning_rate=1e-4), loss = 'categorical_crossentropy', metrics = ['accuracy'])


history = model.fit(X_train, Y_train,
                              validation_data=(X_test,Y_test),
                              batch_size=20,
                              epochs=50,
                              verbose=1)

test_loss, test_acc = model.evaluate(X_test,Y_test)
print('Test loss:', test_loss)
print('Test accuracy:', test_acc)


# Evolución de Accuracy and Loss para el modelo pequeño
# Valores de training y validation
acc      = history.history[ 'accuracy' ]
val_acc  = history.history[ 'val_accuracy' ]
loss     = history.history[    'loss' ]
val_loss = history.history['val_loss' ]

epochs   = range(len(acc)) # Número de epochs

#------------------------------------------------
# Dibujar training y validation accuracy en cada epoch
#------------------------------------------------
plt.plot  ( epochs,     acc )
plt.plot  ( epochs, val_acc )
plt.title ('Training and validation accuracy')
plt.figure()

#------------------------------------------------
# Dibujar training y validation loss en cada epoch
#------------------------------------------------
plt.plot  ( epochs,     loss )
plt.plot  ( epochs, val_loss )
plt.title ('Training and validation loss'   )
#%% TODO: Con dtata augmentation y sin. Prueba de diferentes redes