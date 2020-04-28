import sys
import pickle
import argparse as ap
import numpy as np
import subprocess

from utils import *
from heuristics import *
import net

def get_args():
    p = ap.ArgumentParser()
    
    # Generate or train or test
    p.add_argument("--mode", type=str, required=True, choices=["generate", "train", "test"],
                    help="Operating mode: generate features/labels, train, or test.")
    
    # File names
    p.add_argument("--dataset", type=str, default="dataset.txt",
                    help="Where to save/read dataset")
    p.add_argument("--model", type=str, default="best_model.h5",
                    help="Where to save/read final neural net")
    p.add_argument("--features", type=str, default="features.txt",
                    help="Where to dump/read features")
    p.add_argument("--labels", type=str, default="labels.txt",
                    help="Where to dump/read labels")
    p.add_argument("--params", type=str, default="params.txt",
                    help="Where to save parameters of final model")
    p.add_argument("--evaluation", type=str, default="evaluation.txt",
                    help="Where to save evaluation metrics of final model")

    # Optional arguments
    p.add_argument("--num_instances", type=int, default=25000,
                    help="Number of instances in dataset")
    p.add_argument("--max_boxes", type=int, default=500,
                    help="Max number of boxes per instance")
    p.add_argument("--bin_length", type=int, default=10,
                    help="Max bin length")
    p.add_argument("--bin_width", type=int, default=10,
                    help="Max bin width")
    p.add_argument("--eval_num", type=int, default=500,
                    help="Number of evaluations for hyperparameter search") 

    return p.parse_args()

def generate(args):
    filepath = args.dataset
    generate_raw_dataset(filepath, num_instances=args.num_instances, 
        max_boxes=args.max_boxes, max_bin_length=args.bin_length, max_bin_width=args.bin_width)
    dataset = read_dataset(filepath)
    generate_features(dataset, save=args.features) # generate features
    generate_labels(dataset, save=args.labels) # results from heuristics
    del dataset

def train(args):
    subprocess.Popen(['./tune.sh %s %s %s' %(args.model,args.params,args.eval_num)], shell = True)

def test(args):
    net.test(features_file=args.features, labels_file=args.labels, 
        model_file=args.model, results_file=args.evaluation)

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
