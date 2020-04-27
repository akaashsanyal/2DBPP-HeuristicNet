#!/bin/bash

# File names
TRAINDATA="train_data.txt"
TESTDATA="test_data.txt"
TRAINFEATURES="train_features.txt"
TESTFEATURES="test_features.txt"
TRAINLABELS="train_labels.txt"
TESTLABELS="test_labels.txt"
MODEL="best_model.h5"

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
        
TRAINCMD="python3 main.py \
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
$TRAINCMD

#$TESTGENERATE
#$TESTCMD
