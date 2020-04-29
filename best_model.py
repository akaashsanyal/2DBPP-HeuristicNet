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

model_file = sys.argv[1]

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

X, num_features = pickle.load(open('bigdata/train_features.txt', 'rb'))
labels, num_heuristics = pickle.load(open('bigdata/train_labels.txt', 'rb'))

X = X.astype('float32')

X, labels = data()

activate = 'relu'
first_dense = 32 
first_dropout = {{uniform(0, 1)}}
hidden_dense = 32
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

Y = np_utils.to_categorical(lab_to_correct(labels, first_choice), num_heuristics)

model.fit(X, Y,
          batch_size=128,
          epochs=50,
          verbose=0)

    predictions = model.predict(X_val)
    custom_acc = custom_eval(predictions, val_labels)
    
    score, acc_top3 = model.evaluate(X_val, Y_val, verbose=0)
    


model.save("new_best.h5")


