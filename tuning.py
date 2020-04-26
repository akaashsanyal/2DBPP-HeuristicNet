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


def train(train_feat, train_labels, model_file, epoch_num, first_choice):
    X_train, num_features = pickle.load(open(train_feat, 'rb'))
    labels, num_heuristics = pickle.load(open(train_labels, 'rb'))
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
            
    X_train, X_val, y_train, y_val = train_test_split(X_train, corrects, test_size=0.2, random_state=12345)
    X_train = X_train.astype('float32')
    X_val = X_val.astype('float32')
    Y_train = np_utils.to_categorical(y_train, num_heuristics)
    Y_val = np_utils.to_categorical(y_val, num_heuristics)
    
    
    def model(X_train, Y_train, X_val, Y_val):
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
        model.fit(X_train, Y_train,
                  batch_size=256,
                  nb_epoch=1,
                  verbose=2,
                  validation_data=(X_val, Y_val))
                  
        predictions = model.predict(X_val)
        index_preds = np.argmax(predictions, axis=1)
        corrects = []
        for lab in Y_val:
            winners = np.argwhere(lab == np.min(lab))
            corrects.append(winners.flatten().tolist())
        
        
        score, acc = model.evaluate(X_val, Y_val, verbose=0)
        print('Custom test accuracy:', acc)
        return {'loss': -acc, 'status': STATUS_OK, 'model': model}

    
    
        index_preds = np.argmax(predictions, axis=1)
        
        corrects = []
        for lab in labels:
            winners = np.argwhere(lab == np.min(lab))
            corrects.append(winners.flatten().tolist())
        
        correct_count = 0
        pbar = tqdm(index_preds)
        for p, c in zip(pbar, corrects):
            pbar.set_description("Calculating custom testing accuracy")
            if p in c:
                correct_count+=1
    
    
