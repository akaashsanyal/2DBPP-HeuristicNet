import tensorflow as tf
import numpy as np
import keras
from keras.models import Model, Sequential
from keras.layers import Dropout, Dense, BatchNormalization
from keras.callbacks import ModelCheckpoint
from utils import *
from heuristics import *

filepath = sys.argv[1] 
generate_raw_dataset(filepath, num_instances=4)
dataset = read_dataset(filepath)
features = generate_features(dataset) # generate features
num_features = len(features[0])
labels = [] # results from heuristics
num_heuristics = 10 # depends on how many we do
'''
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
'''
