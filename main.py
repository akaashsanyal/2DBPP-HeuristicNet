import sys
import os
import pickle
import argparse as ap
import numpy as np

from utils import *
from heuristics import *
import net

def get_args():
    p = ap.ArgumentParser()
    
    # Train or test
    p.add_argument("--mode", type=str, required=True, choices=["generate", "train", "test"],
                    help="Operating mode: generate features/labels, train, or test.")
    
    # File names
    p.add_argument("--dataset", type=str, required=True, 
                    help="Where to save/read dataset")
    p.add_argument("--model", type=str, default="my_model.h5",
                    help="Where to save/read final neural net")
    p.add_argument("--features", type=str, default="features.txt",
                    help="Where to dump/read features")
    p.add_argument("--labels", type=str, default="labels.txt",
                    help="Where to dump/read labels")

    return p.parse_args()

def generate(args):
    filepath = args.dataset
    generate_raw_dataset(filepath, num_instances=25000, max_boxes = 500, max_bin_length = 10, max_bin_width = 10)
    dataset = read_dataset(filepath)
    features = generate_features(dataset, save=args.features) # generate features
    num_features = len(features[0])
    labels, num_heuristics = generate_labels(dataset, save=args.labels) # results from heuristics

    return num_features, num_heuristics
    
    '''
features = pickle.load(open("features.txt", 'rb'))

features = pickle.load(open("labels.txt", 'rb'))
'''
    

def train(args):
    net.train(features_file = args.features, labels_file = args.labels, model_file = args.model)



if __name__ == "__main__":
    ARGS = get_args()

    if ARGS.mode.lower() == 'generate':
        generate(ARGS)
    elif ARGS.mode.lower() == 'train':
        train(ARGS)
    elif ARGS.mode.lower() == 'test':
        test(ARGS)
    else:
        raise Exception("Mode given by --mode is unrecognized.")
