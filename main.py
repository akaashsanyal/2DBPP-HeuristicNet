import sys
import os
import pickle
import argparse as ap
import numpy as np
from utils import *
from heuristics import *
from net import *

def get_args():
    p = ap.ArgumentParser()

    # File names
    p.add_argument("--generate", type=str, required=True, choices=["true", "false"],
                    help="Whether to generate dataset or read in existing")
    p.add_argument("--dataset", type=str, required=True, help="Filepath for dataset")
    p.add_argument("--model", type=str, default="my_model.h5",
                    help="Where to store final neural net")
    p.add_argument("--features", type=str,
                    help="Where to dump features")
    p.add_argument("--labels", type=str,
                    help="Where to dump labels")
    
    # Train or test
    p.add_argument("--mode", type=str, required=True, choices=["train", "test"],
                    help="Operating mode: train or test.")

    return p.parse_args()




if __name__ == "__main__":
    ARGS = get_args()
    if ARGS.generate.lower() == 'true':
        generate_raw_dataset(ARGS.dataset, num_instances=25000, max_boxes = 100, max_bin_length = 10, max_bin_width = 10)
        ds = dataset = read_dataset(ARGS.dataset)
        features = generate_features(dataset) # generate features
        num_features = len(features[0])
        labels, num_heuristics = generate_labels(dataset) # results from heuristics
    
    if ARGS.mode.lower() == 'train':
        train(ARGS)
    elif ARGS.mode.lower() == 'test':
        test(ARGS)
    else:
        raise Exception("Mode given by --mode is unrecognized.")
