#!/bin/bash

# File names
TRAINDATA="train_data.txt"
TESTDATA="test_data.txt"
VALDATA="val_data.txt"
TRAINFEATURES="train_features.txt"
TESTFEATURES="test_features.txt"
VALFEATURES="val_features.txt"
TRAINLABELS="train_labels.txt"
TESTLABELS="test_labels.txt"
VALLABELS="val_labels.txt"
MODEL="case2.h5"

# Data generation
TRAININSTANCE="30000"
TESTINSTANCE="1000"
VALINSTANCE="10000"
MAXBOXES="500"
BINLENGTH="20"
BINWIDTH="20"

# Training
EPOCHS="50"

# Testing
CUSTOMEVAL="True"

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
        
VALGENERATE="python3 main.py \
        --mode generate 
        --dataset $VALDATA \
        --features $VALFEATURES \
        --labels $VALLABELS \
        --model $MODEL \
        --num_instances $VALINSTANCE \
        --max_boxes $MAXBOXES \
        --bin_length $BINLENGTH \
        --bin_width $BINWIDTH"

VALCMD="python3 main.py \
        --mode test 
        --dataset $VALDATA \
        --features $VALFEATURES \
        --labels $VALLABELS \
        --model $MODEL \
        --custom_eval $CUSTOMEVAL"

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
        --model $MODEL \
        --custom_eval $CUSTOMEVAL"

$TRAINGENERATE
#$TRAINCMD

$VALGENERATE
#$VALCMD

#$TESTGENERATE
#$TESTCMD
