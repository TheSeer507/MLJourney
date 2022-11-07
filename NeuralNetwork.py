#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf 
from sklearn.compose import make_column_transformer
from sklearn.model_selection import train_test_split


#Create features

X = np.array([-7.0, -4.0, -1.0, 2.0, 5.0, 8.0, 11.0, 14.0])

#Create labels
y = np.array([3.0, 6.0, 9.0, 12.0,15.0,18.0,21.0,24.0])

#Visualize it
print(plt.scatter(X,y))

print(y == X + 10)

def convertToTF(X,y):
    X = tf.constant(X)
    y = tf.constant(y)
    print(X,y)
    return X,y

def modelBuilding(X,y):
    tf.random.set_seed(42)
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(1)
    ])

    model.compile(loss=tf.keras.losses.mae,
                optimizer = tf.keras.optimizers.Adam(),
                metrics=["mae"])
    
    model.fit(tf.expand_dims(X, axis=-1), y, epochs=100)

    print(model.predict([17.0,20.0]))


modelBuilding(X,y)


