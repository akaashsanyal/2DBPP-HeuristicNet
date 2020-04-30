import numpy as np
import pickle
import random
from tqdm import tqdm

import keras
from keras.models import load_model, Sequential
from keras.layers import Dropout, Dense, BatchNormalization, Activation
from keras.metrics import top_k_categorical_accuracy
from keras.utils import np_utils

def top3(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=3) 

def lab_to_correct(labels, first_choice=False):
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

def performance(predictions, labels, results_file):
    lnp = np.asarray(labels)
    best = np.mean(np.min(lnp, axis=1))
    each = np.mean(lnp, axis=0)
    index_preds = np.argmax(predictions, axis=1)
    scores = []
    for lab, ind in zip(labels,index_preds):
        scores.append(lab[ind])
    net = np.mean(scores)

    f = open(results_file, 'a')
    print(f'Average best choice: {round(best,4)}\t\tProportion: {round(best/best,4)}', file=f)
    for i, e in enumerate(each):
        print(f'Average for heuristic {i}: {round(e,4)}\t\tProportion: {round(e/best,4)}', file=f)
    print(f'Average for neural net: {round(net,4)}\t\tProportion: {round(net/best,4)}', file=f)
    f.close()

def train(features_file, labels_file, model_file, epoch_num):
    features, num_features = pickle.load(open(features_file, 'rb'))
    labels, num_heuristics = pickle.load(open(labels_file, 'rb'))
    
    # For labels with same number of bins, choose first occurrence
    # as correct label
    corrects = []
    for lab in labels:
        corrects.append(lab.index(min(lab)))

    one_hot = np.zeros((len(corrects), num_heuristics))
    one_hot[np.arange(len(corrects)),corrects] = 1

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

    model.fit(features, one_hot, epochs=epoch_num, batch_size=32)

    model.save(model_file)

def test(features_file, labels_file, model_file, results_file):
    features, num_features = pickle.load(open(features_file, 'rb'))
    labels, num_heuristics = pickle.load(open(labels_file, 'rb'))


    print(features_file, labels_file, model_file)
    model = load_model(model_file, custom_objects={'top3': top3})
    predictions = model.predict(features)
    custom_acc = custom_eval(predictions, labels)
    print(custom_acc)
    Y = np_utils.to_categorical(lab_to_correct(labels), num_heuristics)
    score, acc_top3 = model.evaluate(features, Y, verbose=0)
    
    f = open(results_file, 'w')
    print('Custom testing accuracy:', custom_acc, file=f)
    print('Top 3 testing accuracy:', acc_top3, file=f)
    f.close()
    performance(predictions, labels, results_file)
