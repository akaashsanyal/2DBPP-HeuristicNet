import tensorflow as tf
import numpy as np
import keras
from keras.models import Model, Sequential
from keras.layers import Dropout, Dense, BatchNormalization
from keras.callbacks import ModelCheckpoint
from utils import *
from heuristics import *

num_features = 30
num_heuristics = 10
data = [] # generate features
labels = [] # results from heuristics


model = Sequential()
model.add(Dense(128, activation='softplus', input_dim=num_features))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(128, activation='softplus'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(128, activation='softplus'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(128, activation='softplus'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(128, activation='softplus'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(num_heuristics, activation='softmax'))

model.compile(optimizer='adamax', \
    loss='categorical_crossentropy', \
    metrics=['accuracy'])

checkpointer = ModelCheckpoint(filepath= 'model_epoch{epoch:02d}.h5', verbose=1)

model.fit(data, labels, epochs=10, batch_size=16, callbacks=[checkpointer])

