#!/bin/bash

# File names
DATADIR="data"
RESULTSDIR="results"

mkdir -p $DATADIR
mkdir -p $RESULTSDIR

TRAINDATA="$DATADIR/train_data.txt"
TESTDATA="$DATADIR/test_data.txt"
TRAINFEATURES="$DATADIR/train_features.txt"
TESTFEATURES="$DATADIR/test_features.txt"
TRAINLABELS="$DATADIR/train_labels.txt"
TESTLABELS="$DATADIR/test_labels.txt"

TUNINGMODEL="$RESULTSDIR/tuning_model.h5"
FINALMODEL="$RESULTSDIR/best_model.h5"
PARAMS="$RESULTSDIR/best_params.txt"
EVALUATION="$RESULTSDIR/evaluation.txt"
PLOT="$RESULTSDIR/accuracy_plot.png"

# Data generation
NUMINSTANCE="2000"
MAXBOXES="200"
BINLENGTH="40"
BINWIDTH="40"
DISTTYPE="1"

# Training
EVALS="400"

# Commands
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
        --bin_width $BINWIDTH \
        --distribution_type $DISTTYPE"

MAINTRAIN="python3 main.py --mode train"

TUNETRAIN="python3 tuning.py \
        --features $TRAINFEATURES \
        --labels $TRAINLABELS \
        --model $TUNINGMODEL \
        --params $PARAMS \
        --evals $EVALS" 

TEST="python3 main.py \
        --mode test \
        --train_dataset $TRAINDATA \
        --train_features $TRAINFEATURES \
        --train_labels $TRAINLABELS \
        --test_dataset $TESTDATA \
        --test_features $TESTFEATURES \
        --test_labels $TESTLABELS \
        --model $FINALMODEL \
        --evaluation $EVALUATION \
        --plot $PLOT \
        --params $PARAMS"

# Run commands
$GENERATE
$MAINTRAIN
$TUNETRAIN
$TEST

