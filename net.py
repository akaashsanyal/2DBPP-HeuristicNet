import tensorflow as tf
import numpy as np
import keras
from keras.models import Model, Sequential
from keras.layers import Dropout, Dense, BatchNormalization
from keras.callbacks import ModelCheckpoint
from utils import *
from heuristics import *

filepath = sys.argv[1] 
print("GENERATING DATA")
generate_raw_dataset(filepath, num_instances=10000, max_boxes = 100, max_bin_length = 10, max_bin_width = 10)
print("READING DATA")
dataset = read_dataset(filepath)
print("GENERATING FEATURES")
features = generate_features(dataset) # generate features
num_features = len(features[0])
print("GENERATING LABELS")
labels, num_heuristics = generate_labels(dataset) # results from heuristics

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

model.fit(data, labels, epochs=10, batch_size=16, callbacks=[checkpointer])

model.save('my_model.h5')
