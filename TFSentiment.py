import tensorflow as tf
import tensorflow_datasets as tfds
import os
import time

start_time = time.time()

dataset, info = tfds.load('imdb_reviews/subwords8k', with_info=True, as_supervised=True)

train_dataset, test_dataset = dataset['train'], dataset['test']

encoder = info.features['text'].encoder

BUFFER_SIZE = 10000
BATCH_SIZE = 64

padded_shapes = ([None], ())

train_dataset = train_dataset.shuffle(BUFFER_SIZE).padded_batch(BATCH_SIZE,
                                                                padded_shapes=padded_shapes)

test_dataset = test_dataset.padded_batch(BATCH_SIZE,
                                padded_shapes=padded_shapes)
"""
model = tf.keras.Sequential([tf.keras.layers.Embedding(encoder.vocab_size, 64),
                             tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
                             tf.keras.layers.Dense(64, activation='relu'),
                             tf.keras.layers.Dense(1, activation='sigmoid')])
model.compile(loss='binary_crossentropy',
              optimizer=tf.keras.optimizers.Adam(1e-4),
              metrics=['accuracy'])

history = model.fit(train_dataset, epochs=10, validation_data=test_dataset,
                   #validation_steps=30)
"""

def pad_to_size(vec, size):
    zeros = [0]*(size-len(vec))
    vec.extend(zeros)
    return vec

def sample_predict(sentence, pad):
    encoded_sample_pred_text = encoder.encode(sentence)
    if pad:
        encoded_sample_pred_text = pad_to_size(encoded_sample_pred_text,64)
    encoded_sample_pred_text = tf.cast(encoded_sample_pred_text, tf.float32)
    predictions = model.predict(tf.expand_dims(encoded_sample_pred_text, 0))

    return predictions

#sample_text = ('This movie was awesome, The Acting was incredible. Highly recommended')
#predictions = sample_predict(sample_text, pad=True, model=model) * 100

#print('Probability this is a positive review %.2f' % predictions)

#sample_text = ('This movie was so so. Didint liked it')
#predictions = sample_predict(sample_text, pad=True, model=model) * 100

#print('Probability this is a positive review %.2f' % predictions)

model = tf.keras.Sequential([tf.keras.layers.Embedding(encoder.vocab_size, 64),
                            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
                            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
                            tf.keras.layers.Dense(64, activation='relu'),
                            tf.keras.layers.Dropout(0.5),
                            tf.keras.layers.Dense(1, activation='sigmoid')])
model.compile(loss='binary_crossentropy',
              optimizer=tf.keras.optimizers.Adam(1e-4),
              metrics=['accuracy'])

history = model.fit(train_dataset, epochs=15, validation_data=test_dataset,
                   validation_steps=30)

sample_text = ('Great Service, Incident was resolved promptly. Excellent')
predictions = sample_predict(sample_text, pad=True) * 100

print('Probability this is a positive review %.2f' % predictions)

sample_text = ('Bad Service, no follow up, closed righ away. Please improve your service')
predictions = sample_predict(sample_text, pad=True) * 100

print('Probability this is a positive review %.2f' % predictions)
print("--- %s seconds ---" % (time.time() - start_time))

