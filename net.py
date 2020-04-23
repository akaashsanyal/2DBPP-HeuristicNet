import numpy as np
import pickle
from tqdm import tqdm
from keras.models import load_model, Sequential
from keras.layers import Dropout, Dense, BatchNormalization

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

def test(features_file, labels_file, model_file, custom=False):
    features, num_features = pickle.load(open(features_file, 'rb'))
    labels, num_heuristics = pickle.load(open(labels_file, 'rb'))
    
    model = load_model(model_file)
    
    if custom:
        predictions = model.predict(features)
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
        
        print(f"Custom testing accuracy: {100*correct_count/len(index_preds):.2f}%")


    else:
        # For labels with same number of bins, choose first occurrence
        # as correct label
        corrects = []
        for lab in labels:
            corrects.append(lab.index(min(lab)))

        one_hot = np.zeros((len(corrects), num_heuristics))
        one_hot[np.arange(len(corrects)),corrects] = 1

        results = model.evaluate(features, one_hot, batch_size=32)
        print('test loss, test acc:', results)
