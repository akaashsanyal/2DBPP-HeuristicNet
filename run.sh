#!/bin/bash

# File names
DATADIR="tempdata"
RESULTSDIR="tempresults"

mkdir -p $DATADIR
mkdir -p $RESULTSDIR

TRAINDATA="$DATADIR/train_data.txt"
TESTDATA="$DATADIR/test_data.txt"
TRAINFEATURES="$DATADIR/train_features.txt"
TESTFEATURES="$DATADIR/test_features.txt"
TRAINLABELS="$DATADIR/train_labels.txt"
TESTLABELS="$DATADIR/test_labels.txt"

MODEL="$RESULTSDIR/best_model.h5"
PARAMS="$RESULTSDIR/best_params.txt"
LOGFILE="$RESULTSDIR/tuning_log.txt"
#EVALUATION="bigresults/evaluation.txt"
EVALUATION="$RESULTSDIR/temp_eval.txt"

# Data generation
NUMINSTANCE="100"
TRAININSTANCE="30000"
TESTINSTANCE="10000"
MAXBOXES="1500"
BINLENGTH="40"
BINWIDTH="40"

# Training
EVALS="10"

TRAINCMD="python3 main.py \
        --mode train 
        --dataset $TRAINDATA \
        --features $TRAINFEATURES \
        --labels $TRAINLABELS \
        --model $MODEL \
        --params $PARAMS \
        --eval_num $EVALS"
        
TESTCMD="python3 main.py \
        --mode test 
        --dataset $TESTDATA \
        --features $TESTFEATURES \
        --labels $TESTLABELS \
        --model $MODEL \
        --evaluation $EVALUATION"

TUNETRAIN="python3 tuning.py \
        --features $TRAINFEATURES \
        --labels $TRAINLABELS \
        --model $MODEL \
        --params $PARAMS \
        --evals $EVALS" 

GENERATE="python3 main.py \
        --mode generate \
        --train_dataset $TRAINDATA \
        --train_features $TRAINFEATURES \
        --train_labels $TRAINLABELS \
        --test_dataset $TESTDATA \
        --test_features $TESTFEATURES \
        --test_labels $TESTLABELS \
        --num_instances $NUMINSTANCE \
        --max_boxes $MAXBOXES \
        --bin_length $BINLENGTH \
        --bin_width $BINWIDTH"

MAINTRAIN="python3 main.py --mode train"

#$GENERATE
#$MAINTRAIN
$TUNETRAIN

#$NEWTRAIN
#$TESTCMD

#$TRAINCMD
#$TESTCMD
