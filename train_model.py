import os
from tensorflow.keras.callbacks import TensorBoard

import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

from tensorflow.keras.optimizers import Adam

from model import build_model
from preprocess_data import load_data

log_dir = os.path.join(r'C:\Users\Artatrana\Downloads\action_detection\data_collection\Logs')
tb_callback = TensorBoard(log_dir=log_dir)

actions = np.array(['hello','thank you','yes', 'no', 'how are you', 'fine', 'goodbye'])

model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,1662)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))

X_train, X_test, y_train, y_test = load_data()
model = build_model(y_train.shape[1])

initial_lr = 0.0001
optimizer = Adam(learning_rate=initial_lr)

model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['categorical_accuracy'])
model.fit(X_train, y_train, epochs=1500, callbacks=[tb_callback])

#model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
#model.fit(X_train, y_train, epochs=1000, callbacks=[tb_callback])

model.save('action.h5')
