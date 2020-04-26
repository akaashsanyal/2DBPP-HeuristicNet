import numpy as np
import pickle
from tqdm import tqdm
import random

from sklearn.model_selection import train_test_split

import keras
from keras.models import load_model, Sequential
from keras.layers import Dropout, Dense, BatchNormalization, Activation
from keras.utils import np_utils

from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform

def lab_to_correct(labels, first_choice=True):
    if first_choice:
        # For labels with same number of bins, choose first occurrence
        # as correct label
        corrects = []
        for lab in labels:
            corrects.append(lab.index(min(lab)))
    else:
        # For labels with same number of bins, choose random occurrence
        # as correct label
        corrects = []
        for lab in labels:
            m = min(lab)
            min_indices = [i for i, x in enumerate(lab) if x == m]
            corrects.append(random.choice(min_indices))

    return corrects

def custom_eval(predictions, labels):
    index_preds = np.argmax(predictions, axis=1)
    corrects = []
    correct_count = 0
    for lab in labels:
        winners = np.argwhere(lab == np.min(lab))
        corrects.append(winners.flatten().tolist())
    for p, c in zip(index_preds, corrects):
        if p in c:
            correct_count+=1
    
    return correct_count/len(index_preds)

def model(X_train, train_labels, X_val, val_labels):
    model = Sequential()
    model.add(Dense({{choice([32, 64, 128])}}, input_dim=num_features))
    model.add(Activation({{choice(['relu', 'softplus'])}}))
    model.add(BatchNormalization())
    model.add(Dropout({{uniform(0, 1)}}))
    model.add(Dense({{choice([32, 64, 128])}}))
    model.add(Activation({{choice(['relu', 'softplus'])}}))
    model.add(BatchNormalization())
    model.add(Dropout({{uniform(0, 1)}}))

    if {{choice(['two', 'three'])}} == 'three':
        model.add(Dense({{choice([32, 64, 128])}}))
        model.add(Activation({{choice(['relu', 'softplus'])}}))
        model.add(BatchNormalization())
        model.add(Dropout({{uniform(0, 1)}}))
        if {{choice(['three', 'four'])}} == 'four':
            model.add(Dense({{choice([32, 64, 128])}}))
            model.add(Activation({{choice(['relu', 'softplus'])}}))
            model.add(BatchNormalization())
            model.add(Dropout({{uniform(0, 1)}}))

    model.add(Dense(num_heuristics, activation='softmax'))

    adam = keras.optimizers.Adam(lr={{choice([10**-3, 10**-2, 10**-1])}})
    rmsprop = keras.optimizers.RMSprop(lr={{choice([10**-3, 10**-2, 10**-1])}})

    choiceval = {{choice(['adam', 'rmsprop'])}}
    if choiceval == 'adam':
        optim = adam
    else:
        optim = rmsprop

    model.compile(loss='categorical_crossentropy', metrics=['accuracy'],optimizer=optim)
    
    Y_train = np_utils.to_categorical(lab_to_correct(train_labels), num_heuristics)
    Y_val = np_utils.to_categorical(lab_to_correct(val_labels), num_heuristics)
    
    model.fit(X_train, Y_train,
              batch_size=256,
              nb_epoch=30,
              verbose=2,
              validation_data=(X_val, Y_val))

    predictions = model.predict(X_val)
    custom_acc = custom_eval(predictions, val_labels)
    
    score, acc = model.evaluate(X_val, Y_val, verbose=0)
    print('Custom test accuracy:', custom_acc)
    
    return {'loss': -custom_acc, 'status': STATUS_OK, 'model': model}

def data():
    features, num_features = pickle.load(open('train_features.txt', 'rb'))
    labels, num_heuristics = pickle.load(open('train_labels.txt', 'rb'))

    X_train, X_val, train_labels, val_labels = train_test_split(features, labels, test_size=0.2, random_state=12345)
    X_train = X_train.astype('float32')
    X_val = X_val.astype('float32')

    return X_train, train_labels, X_val, val_labels

best_run, best_model = optim.minimize(model=model,
                                      data=data,
                                      algo=tpe.suggest,
                                      max_evals=10,
                                      functions=[custom_eval,lab_to_correct],
                                      trials=Trials())

best_model.save("best_model.h5")
print(best_run)
