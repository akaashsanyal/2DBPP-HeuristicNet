import tensorflow as tf
import numpy as np
import pickle
import keras
from keras.models import Model, Sequential
from keras.layers import Dropout, Dense, BatchNormalization
from keras.callbacks import ModelCheckpoint
from utils import *
from heuristics import *

filepath = sys.argv[1] 
#generate_raw_dataset(filepath, num_instances=25000, max_boxes = 500, max_bin_length = 10, max_bin_width = 10)
dataset = read_dataset(filepath)
#features = generate_features(dataset, save="features.txt") # generate features
#'''
with open ("features.txt", 'rb') as fp:
    features = pickle.load(fp)
#'''
num_features = len(features[0])
labels, num_heuristics = generate_labels(dataset, save="labels_nonrandom.txt") # results from heuristics
'''
with open ("labels.txt", 'rb') as fp:
    labels = pickle.load(fp)
num_heuristics = labels.size
'''

print("STARTING NEURAL NETWORK")
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

model.fit(features, labels, epochs=50, batch_size=32)

model.save('nonrandom_model.h5')
