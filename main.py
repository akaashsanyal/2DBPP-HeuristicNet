import sys
import pickle
import argparse as ap
import numpy as np

from utils import *
from heuristics import *
import best_model

def get_args():
    p = ap.ArgumentParser()
    
    # Generate or train or test
    p.add_argument("--mode", type=str, required=True, choices=["generate", "train", "test"],
                    help="Operating mode: generate features/labels, train, or test.")
    
    # File names
    p.add_argument("--model", type=str, default="best_model.h5",
                    help="Where to save/read final neural net")
    
    p.add_argument("--train_dataset", type=str, default="train_dataset.txt",
                    help="Where to save/read train dataset")
    p.add_argument("--train_features", type=str, default="train_features.txt",
                    help="Where to dump/read train features")
    p.add_argument("--train_labels", type=str, default="train_labels.txt",
                    help="Where to dump/read train labels")
    p.add_argument("--test_dataset", type=str, default="test_dataset.txt",
                    help="Where to save/read test dataset")
    p.add_argument("--test_features", type=str, default="test_features.txt",
                    help="Where to dump/read test features")
    p.add_argument("--test_labels", type=str, default="test_labels.txt",
                    help="Where to dump/read test labels")
    
    p.add_argument("--params", type=str, default="params.txt",
                    help="Where to save parameters of final model")
    p.add_argument("--evaluation", type=str, default="evaluation.txt",
                    help="Where to save evaluation metrics of final model")
    p.add_argument("--plot", type=str, default="accuracy_plot.png",
                    help="Where to save accuracy plot")

    # Optional arguments
    p.add_argument("--num_instances", type=int, default=40000,
                    help="Number of instances in dataset")
    p.add_argument("--max_boxes", type=int, default=1000,
                    help="Max number of boxes per instance")
    p.add_argument("--bin_length", type=int, default=40,
                    help="Max bin length")
    p.add_argument("--bin_width", type=int, default=40,
                    help="Max bin width")
    p.add_argument("--eval_num", type=int, default=500,
                    help="Number of evaluations for hyperparameter search") 

    return p.parse_args()

def generate(args):
    train_filepath = args.train_dataset
    generate_raw_dataset(train_filepath, num_instances=int(args.num_instances*3/4), 
        max_boxes=args.max_boxes, max_bin_length=args.bin_length, max_bin_width=args.bin_width)
    train_dataset = read_dataset(train_filepath)
    generate_features(train_dataset, save=args.train_features) # generate features
    generate_labels(train_dataset, save=args.train_labels) # results from heuristics
    del train_dataset
    
    test_filepath = args.train_dataset
    generate_raw_dataset(test_filepath, num_instances=int(args.num_instances*1/4), 
        max_boxes=args.max_boxes, max_bin_length=args.bin_length, max_bin_width=args.bin_width)
    test_dataset = read_dataset(test_filepath)
    generate_features(test_dataset, save=args.test_features) # generate features
    generate_labels(test_dataset, save=args.test_labels) # results from heuristics
    del test_dataset

def train(args):
    print("Please run tune.sh to train and tune hyperparameters")

def test(args):
    best = input("Is best_model.py updated with the best parameters? (y/n)   ")
    if best.lower() == 'y':
        best_model.test(args.train_features, args.train_labels, args.test_features, 
            args.test_labels, args.model, args.evaluation, args.plot)
    else:
        print('Update the file before testing')
    
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
