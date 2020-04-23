import tensorflow as tf
import numpy as np
import pickle
import keras
from keras.models import load_model, Sequential
from keras.layers import Dropout, Dense, BatchNormalization

def train(features_file, labels_file, model_file, epoch_num):
    features, num_features = pickle.load(open(features_file, 'rb'))
    labels, num_heuristics = pickle.load(open(labels_file, 'rb'))
    
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

    model.fit(features, labels, epochs=epoch_num, batch_size=32)

    model.save(model_file)

def test(features_file, labels_file, model_file, custom=False):
    features, num_features = pickle.load(open(features_file, 'rb'))
    labels, num_heuristics = pickle.load(open(labels_file, 'rb'))

    model = load_model(model_file)
    if custom:
        print("Not implemented")
    else:
        results = model.evaluate(features, labels, batch_size=32)
        print('test loss, test acc:', results)
