#!/bin/bash

# In general, you should do fast runs; we don't need to converge fully
# unless we want to make the claim that the net succeeds w.r.t. all
# architecture choices (pretty much proven in the message varying case, less
# clear when varying net encoding)

set -e

# 1. Fix the 256-dimensional hidden message, and vary message size m.
for i in 2 5 64 128 900; do
    for k in 1; do
    # for k in 1 2 3 4 5; do  # TODO: Eventually run many.
        python3 -u rnn_syn.py --components \
            --epochs 5000 \
            --model end2end \
            --msgs_file "data/arch_msg_test_5000_ncomm$i.$k.pkl" \
            --batch_size 128 \
            --asym_max_images 5 \
            --asym_min_targets 2 \
            --asym_min_distractors 1 \
            --save_max 1024 \
            --n_comm "$i" \
            --n_hidden 256 | \
            tee "saves/arch_msg_output_$i.$k"
    done
done

# 2. Fix the message size m to be "large" (1024) and vary n_hidden.
# NOTE: People might not think whatever m I choose is BIG enough - notice that
# we're not saying that the networks will never result in a degenreate solution
# in general; saying that even with (fairly substantial) bandwith, the emergent
# protocol is fairly simple
for i in 5 64 128 900; do
    for k in 1; do
    # for k in 1 2 3 4 5; do  # TODO: Eventually run many.
        python3 -u rnn_syn.py --components \
            --epochs 5000 \
            --model end2end \
            --msgs_file "data/arch_rnn_test_5000_nhidden$i.$k.pkl" \
            --batch_size 128 \
            --asym_max_images 5 \
            --asym_min_targets 2 \
            --asym_min_distractors 1 \
            --save_max 1024 \
            --n_comm 1024 \
            --n_hidden "$i" | \
            tee "saves/arch_rnn_output_$i.$k"
    done
done

# TODO: One thing I haven't done yet is vary CNN encoding size, but it's always
# as big as (or bigger) than the message sizes we try, if that's any consolation

# Finally, try the smallest model
python3 -u rnn_syn.py --components \
    --epochs 5000 \
    --model end2end \
    --msgs_file "data/arch_small_test_5000.pkl" \
    --batch_size 128 \
    --asym_max_images 5 \
    --asym_min_targets 2 \
    --asym_min_distractors 1 \
    --save_max 1024 \
    --n_comm 2 \
    --n_hidden 2 | \
    tee "saves/arch_small_output"
