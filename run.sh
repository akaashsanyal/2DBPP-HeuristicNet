#!/bin/bash

# File names
TRAINDATA="data/train_data.txt"
TESTDATA="data/test_data.txt"
TRAINFEATURES="bigdata/train_features.txt"
TESTFEATURES="bigdata/test_features.txt"
TRAINLABELS="bigdata/train_labels.txt"
TESTLABELS="bigdata/test_labels.txt"
MODEL="bigresults/best_model.h5"
PARAMS="bigresults/best_params.txt"
#EVALUATION="bigresults/evaluation.txt"
EVALUATION='temp_eval.txt'

# Data generation
TRAININSTANCE="30000"
TESTINSTANCE="10000"
MAXBOXES="1500"
BINLENGTH="40"
BINWIDTH="40"

# Training
EVALS="400"

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
        --params $PARAMS \
        --eval_num $EVALS"
        
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
        --evaluation $EVALUATION"

NEWTRAIN="python3 tuning.py $MODEL $PARAMS $EVALS" 

#$TRAINGENERATE
#$TESTGENERATE

#$NEWTRAIN
$TESTCMD

#$TRAINCMD
#$TESTCMD
