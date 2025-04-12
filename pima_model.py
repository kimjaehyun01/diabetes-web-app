import sys
assert sys.version_info >= (3, 5)

import sklearn
assert sklearn.__version__ >= "0.20"

import tensorflow as tf
from tensorflow import keras
assert tf.__version__ >= "2.0"

import numpy as np
import os
import pandas as pd
from keras.layers import Dense
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

np.random.seed(42)
tf.random.set_seed(42)

import matplotlib as mpl
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

data = pd.read_csv('./diabetes.csv', sep=',')

print("\ndata.head(): \n",
      data.head())

data.describe()

data.info()

print("\n\nStep 2 - Prepare the data for the model building")
X = data.values[:,0:8]
y = data.values[:,8]

scaler = MinMaxScaler()
scaler.fit(X)
X = scaler.transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

print("\n\nStep 3 - Create and train the model")

inputs = keras.Input(shape=(8,))
hidden1 = Dense(12, activation='relu')(inputs)
hidden2 = Dense(8, activation='relu')(hidden1)
output = Dense(1, activation='sigmoid')(hidden2)
model = keras.Model(inputs, output)

model.summary()
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=30, batch_size=16, verbose=0)

fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('epochs')
ax1.set_ylabel('loss', color=color)
ax1.plot(history.history['loss'], color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()

color = 'tab:blue'
ax2.set_ylabel('accuracy', color=color)
ax2.plot(history.history['accuracy'], color=color)
ax2.tick_params(axis='y', labelcolor = color)

fig.tight_layout()
plt.show()

X_new = X_test[:3]
print("\nnp.round(model.predict(X_new), 2): \n",
      np.round(model.predict(X_new), 2))

print("\nExporting SaveModels: ")

model.save('pima model.keras')

model = keras.models.load_model('pima_model.keras')

X_new = X_test[:3]
print("\nnp.round(model.predict(X_new), 2): \n",
      np.round(model.predict(X_new), 2))
