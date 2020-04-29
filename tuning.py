import numpy as np
import pickle
import random
import sys

from sklearn.model_selection import train_test_split

import keras
from keras.models import load_model, Sequential
from keras.layers import Dropout, Dense, BatchNormalization, Activation
from keras.utils import np_utils
from keras.metrics import top_k_categorical_accuracy

from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform

model_file = sys.argv[1]
param_file = sys.argv[2]
eval_num = int(sys.argv[3])

def top3(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=3) 

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
    activate = {{choice(['relu', 'softplus'])}}
    first_dense = {{choice([32, 64, 128])}}
    first_dropout = {{uniform(0, 1)}}
    hidden_dense = {{choice([32, 64, 128])}}
    hidden_dropout = {{uniform(0, 1)}}
    num_hidden = {{choice([0,1,2,3])}}

    model = Sequential()
    model.add(Dense(first_dense, input_dim=num_features))
    model.add(Activation(activate))
    model.add(BatchNormalization())
    model.add(Dropout(first_dropout))
    
    if num_hidden != 0:
        model.add(Dense(hidden_dense))
        model.add(Activation(activate))
        model.add(BatchNormalization())
        model.add(Dropout(hidden_dropout))    
        if num_hidden != 1:
            model.add(Dense(hidden_dense))
            model.add(Activation(activate))
            model.add(BatchNormalization())
            model.add(Dropout(hidden_dropout))
            if num_hidden != 2:
                model.add(Dense(hidden_dense))
                model.add(Activation(activate))
                model.add(BatchNormalization())
                model.add(Dropout(hidden_dropout))
    
    model.add(Dense(num_heuristics, activation='softmax'))

    adam = keras.optimizers.Adam(lr={{choice([10**-3, 10**-2, 10**-1])}})

    model.compile(loss='categorical_crossentropy', metrics = [top3], optimizer=adam)
    
    first_choice = False 
    
    Y_train = np_utils.to_categorical(lab_to_correct(train_labels, first_choice), num_heuristics)
    Y_val = np_utils.to_categorical(lab_to_correct(val_labels, first_choice), num_heuristics)
    
    model.fit(X_train, Y_train,
              batch_size=128,
              epochs=50,
              verbose=0,
              validation_data=(X_val, Y_val))

    predictions = model.predict(X_val)
    custom_acc = custom_eval(predictions, val_labels)
    
    score, acc_top3 = model.evaluate(X_val, Y_val, verbose=0)
    
    log_file = open('bigresults/log_file.txt', 'a')
    print('Custom validation accuracy:', custom_acc, file = log_file)
    print('Top 3 validation accuracy:', acc_top3, file = log_file)
    print('_________________________', file = log_file)
    log_file.close()
    return {'loss': -custom_acc, 'status': STATUS_OK, 'model': model}

def data():
    features, num_features = pickle.load(open('bigdata/train_features.txt', 'rb'))
    labels, num_heuristics = pickle.load(open('bigdata/train_labels.txt', 'rb'))

    X_train, X_val, train_labels, val_labels = train_test_split(features, labels, test_size=0.2, random_state=12345)
    X_train = X_train.astype('float32')
    X_val = X_val.astype('float32')

    return X_train, train_labels, X_val, val_labels

best_run, best_model = optim.minimize(model=model,
                                      data=data,
                                      algo=tpe.suggest,
                                      max_evals=eval_num,
                                      eval_space=True,
                                      functions=[custom_eval,lab_to_correct,top3],
                                      trials=Trials())
best_model.save(model_file)
f = open(param_file, 'w') 
print(best_run, file = f)
f.close()

