import tensorflow as tf
import numpy as np
import pickle
import keras
from keras.models import Model, Sequential
from keras.layers import Dropout, Dense, BatchNormalization
from keras.callbacks import ModelCheckpoint
from utils import *
from heuristics import *

def train(features_file, labels_file, model_file):
    features, num_features = pickle.load(open("features.txt", 'rb'))
    labels, num_heuristics = pickle.load(open("labels.txt", 'rb'))
    
    print("TRAINING NEURAL NETWORK")
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

    model.save(model_file)
