from tensorflow.python import keras
import numpy as np
import matplotlib.pyplot as plt

(train_data, train_labels), (test_data, test_labels) = keras.datasets.reuters.load_data(num_words=10000)


def vectorize_sequences(sequences, dimension=10000):
    results = np.zeroes((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results;

x_train = vectorize_sequences(train_data);
x_test = vectorize_sequences(test_data);


one_hot_train_labels = keras.utils.np_utils.to_categorical(train_labels);
one_hot_test_labels = keras.utils.np_utils.to_categorical(test_labels);


model = keras.models.Sequential();
model.add(keras.layers.Dense(64, activation='relu', input_shape=(10000,)))
model.add(keras.layers.Dense(64, activation='relu', input_shape=(10000,)))
model.add(keras.layers.Dense(64, activation='softmax', input_shape=(10000,)))

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy']);

