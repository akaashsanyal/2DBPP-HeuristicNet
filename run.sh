#!/bin/bash

# File names
TRAINDATA="data/train_data.txt"
TESTDATA="data/test_data.txt"
TRAINFEATURES="data/train_features.txt"
TESTFEATURES="data/test_features.txt"
TRAINLABELS="data/train_labels.txt"
TESTLABELS="data/test_labels.txt"
MODEL="results/best_model.h5"

# Data generation
TRAININSTANCE="30000"
TESTINSTANCE="10000"
MAXBOXES="500"
BINLENGTH="20"
BINWIDTH="20"

# Training
EPOCHS="50"

TRAINGENERATE="python3 main.py \
        --mode generate 
        --dataset $TRAINDATA \
        --features $TRAINFEATURES \
        --labels $TRAINLABELS \
        --model $MODEL \
        --num_instances $TRAININSTANCE \
        --max_boxes $MAXBOXES \
        --bin_length $BINLENGTH \
        --bin_width $BINWIDTH"
        
OLDTRAINCMD="python3 main.py \
        --mode train 
        --dataset $TRAINDATA \
        --features $TRAINFEATURES \
        --labels $TRAINLABELS \
        --model $MODEL \
        --epochs $EPOCHS"
        
TESTGENERATE="python3 main.py \
        --mode generate 
        --dataset $TESTDATA \
        --features $TESTFEATURES \
        --labels $TESTLABELS \
        --model $MODEL \
        --num_instances $TESTINSTANCE \
        --max_boxes $MAXBOXES \
        --bin_length $BINLENGTH \
        --bin_width $BINWIDTH"
        
TESTCMD="python3 main.py \
        --mode test 
        --dataset $TESTDATA \
        --features $TESTFEATURES \
        --labels $TESTLABELS \
        --model $MODEL"

#$TRAINGENERATE
#$OLDTRAINCMD

#$TESTGENERATE
#$TESTCMD
