import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
x_train.shape

y_train[:3]


y_test = y_test.reshape(-1,)

resim_siniflari = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]

def plot_sample (x, y, index):
    plt.figure(figsize = (15,2))
    plt.imshow(x[index])
    plt.xlabel(resim_siniflari[y[index]])
    
plot_sample(x_test, y_test, 0)

plot_sample(x_test, y_test, 24)

#Normalizasyon 

x_train = x_train / 255
x_test = x_test / 255

#CNN 

deep_learning_model = models.Sequential([
    #Konvolüsyon Katmanı
    layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    #Yukarıdaki özelliklere ve training bilgillerine göre modeli eğittim
    layers.Flatten(), #CNN ile otomatik bağlıyor
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
    ])

deep_learning_model.compile(optimizer='adam',
                            loss='sparse_categorical_crossentropy',  #Çıktılar direk rakam değil truck için 000000000001 
                            metrics=['accuracy'])


deep_learning_model.fit(x_train, y_train, epochs=5)


deep_learning_model.evaluate(x_test, y_test)

y_pred = deep_learning_model(x_test)

y_pred[:3]






























































