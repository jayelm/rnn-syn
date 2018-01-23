#!/bin/bash

# Large multi-image (i.e. 4 options) communication game
python3 rnn-syn.py --data data/500_400 --model end2end --test --epochs 100 --n_comm 128
