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
import matplotlib.pyplot as plt

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


def test(train_features_f, train_labels_f, test_features_f, test_labels_f, model_file, results_file, plot_file):
    test_X, num_features = pickle.load(open(test_features_f, 'rb'))
    test_labels, num_heuristics = pickle.load(open(test_labels_f, 'rb'))
    
    features, num_features = pickle.load(open(train_features_f, 'rb'))
    labels, num_heuristics = pickle.load(open(train_labels_f, 'rb'))
    
    X_train, X_val, train_labels, val_labels = train_test_split(features, labels, test_size=0.2, random_state=12345)
    X_train = X_train.astype('float32')
    X_val = X_val.astype('float32')
    
    activate = 'relu'
    first_dense = 32 
    first_dropout = 0.04697373821486008
    hidden_dense = 32
    hidden_dropout = 0.02213872171606842

    model = Sequential()
    model.add(Dense(first_dense, input_dim=num_features))
    model.add(Activation(activate))
    model.add(BatchNormalization())
    model.add(Dropout(first_dropout))

    model.add(Dense(hidden_dense))
    model.add(Activation(activate))
    model.add(BatchNormalization())
    model.add(Dropout(hidden_dropout))    

    model.add(Dense(hidden_dense))
    model.add(Activation(activate))
    model.add(BatchNormalization())
    model.add(Dropout(hidden_dropout))

    model.add(Dense(num_heuristics, activation='softmax'))

    adam = keras.optimizers.Adam(lr=0.01)

    model.compile(loss='categorical_crossentropy', metrics = [top3], optimizer=adam)

    first_choice = False 

    Y_train = np_utils.to_categorical(lab_to_correct(train_labels, first_choice), num_heuristics)
    Y_val = np_utils.to_categorical(lab_to_correct(val_labels, first_choice), num_heuristics)

    history = model.fit(X_train, Y_train,
              batch_size=128,
              epochs=100,
              verbose=2,
              validation_data=(X_val, Y_val))

    predictions = model.predict(test_X)
    custom_acc = custom_eval(predictions, test_labels)
    Y_test = np_utils.to_categorical(lab_to_correct(test_labels, first_choice), num_heuristics)
    score, acc_top3 = model.evaluate(test_X, Y_test, verbose=1)

    f = open(results_file, 'w')
    print('Custom testing accuracy:', custom_acc, file=f)
    print('Top 3 testing accuracy:', acc_top3, file=f)
    f.close()

    performance(predictions, test_labels, results_file)

    plt.plot(history.history['top3'], label='Top 3 Accuracy (testing data)')
    plt.plot(history.history['val_top3'], label='Top 3 Accuracy (validation data)')
    plt.title('Top 3 Accuracy vs. Epoch')
    plt.ylabel('Top 3 Accuracy')
    plt.xlabel('Epoch')
    plt.legend(loc="best")
    plt.savefig(plot_file)

    model.save(model_file)
