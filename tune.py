import numpy as np
import pickle
from tqdm import tqdm
from keras.models import load_model, Sequential
from keras.layers import Dropout, Dense, BatchNormalization
import talos

x, num_features = pickle.load(open("train_features.txt", 'rb'))
labels, num_heuristics = pickle.load(open("train_labels.txt", 'rb'))

x_val, _ = pickle.load(open("val_features.txt", 'rb'))
labels_val, _ = pickle.load(open("val_labels.txt", 'rb'))

# For labels with same number of bins, choose first occurrence
# as correct label
first_corrects = []
first_corrects_val = []
for lab in labels:
    first_corrects.append(lab.index(min(lab)))
for lab in labels_val:
    first_corrects_val.append(lab.index(min(lab)))

y = np.zeros((len(first_corrects), num_heuristics))
y[np.arange(len(first_corrects)),first_corrects] = 1

y_val = np.zeros((len(first_corrects_val), num_heuristics))
y_val[np.arange(len(first_corrects_val)),first_corrects_val] = 1

def tune_model(x_train, y_train, x_val, y_val, params):

    model = Sequential()
    model.add(Dense(params['first_neuron'], input_dim=num_features, activation=params['activation']))
    model.add(BatchNormalization())
    model.add(Dropout(params['dropout']))
    for i in range(params['hidden_layers']):
        model.add(Dense(params['first_neuron'],activation=params['activation']))
        model.add(BatchNormalization())
        model.add(Dropout(params['dropout']))
    model.add(Dense(num_heuristics, activation='softmax'))

    model.compile(loss=params['losses'],
                  optimizer=params['optimizer'],
                  metrics=['acc', talos.utils.metrics.f1score])
    
    history = model.fit(x_train, y_train, 
                        validation_data=[x_val, y_val],
                        batch_size=params['batch_size'],
                        callbacks=[talos.utils.live()],
                        epochs=params['epochs'],
                        verbose=0)

    return history, model

p = {'activation':['softplus', 'relu'],
     'optimizer': ['adamax'],
     'losses': ['categorical_crossentropy'],
     'shapes': ['brick'],          
     'first_neuron': [32, 64, 128],     
     'hidden_layers':[1, 2, 3],
     'dropout': [.2, .3, .5], 
     'batch_size': [32],
     'epochs': [100]}

scan_object = talos.Scan(x, y, model=tune_model, params=p, experiment_name='tuning', fraction_limit=0.1)
