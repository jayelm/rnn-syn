#!/bin/bash

# Recipes for experiments.
# Note: these recipes don't report generalization statistics (i.e. accuracy
# on test data). TODO: Implement that, while still enabling saving test/train
# (maybe save 2 versions of messages: 1 -train, 1 -test)

# Maximum number of messages to save
SAVE_MAX=1000

set -e

# 500_200: 3 pick 2 game
# 500_400: 8 pick 4 game

for data in 500_200 500_400; do
    for model in feature end2end; do
        for n_comm in 2 4 64; do
            if [ "$n_comm" = "64" ]; then
                # Continuous
                comm_type=continuous
            else
                comm_type=discrete
            fi
            echo $data $model $n_comm $comm_type
            python3 rnn-syn.py --data "data/$data" --model "$model" --epochs 25 --n_comm "$n_comm" --comm_type "$comm_type" --save_max "$SAVE_MAX"
        done
    done
done
