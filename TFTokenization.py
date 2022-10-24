import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
import tensorflow_datasets as tfds
import pandas as pd
import numpy as np
import re
import html2text as ht
from tensorflow.keras import layers
from tensorflow import keras
import matplotlib.pyplot as plt

RootFolder = r'C:\MachineLearningModels\TensorflowTesting'
DataFilePath = RootFolder + '\\DataFiles\\'
OutputFilePath = RootFolder + '\\OutputFiles\\'

IncidentTokenized = pd.read_csv(DataFilePath + '2020_incidents_Details_Input.csv')

print(IncidentTokenized.head())

sentences = [
    'Issues with Microsoft Office',
    'Microsoft Excel has issues',
    'Microsoft Office is not Working!'
]

tokenizer = Tokenizer(num_words= None, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                      lower=True, split=' ', char_level=False, oov_token=None)
tokenizer.fit_on_texts(IncidentTokenized['IncidentFullClean'])
model = tf.keras.Sequential()
model.add(tf.keras.layers.Embedding(1000, 64, input_length=10))
model.compile(optimizer='adam', loss='binary_crossentropy')
IncidentTokenized['Tokens'] = tokenizer.texts_to_sequences(IncidentTokenized['IncidentFullClean'])
print(IncidentTokenized)
IncidentTokenized.to_csv(OutputFilePath + 'IncidentDetailsTokenized.csv')
#word_index = tokenizer.word_index
#print(word_index)

