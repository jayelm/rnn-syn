# Learning compositional communication protocols in end-to-end reference games

## Model training

To generate a spatial relations dataset, use `swdata.py`:

```bash
python swdata.py --n_configs 15 --samples_each_config 100 \
    --n_targets 2 --n_distractors 1 --n_cpu 1
```

Each "configuration" is a unique target/distractor pair with a unique spatial
relation (`x` or `y`, `left/up` or `down/right`).

This creates a folder, `data/{num_configs}_{samples_each_config}`, with a
`pkl.gz` file for each configuration. Metadata for the dataset is located in
`metadata.json`.

Then run

```bash
python rnn-syn.py --data data/15_100 --test --model end2end
```

to train an end-to-end model on some configurations and report test accuracy on
unseen configurations with a 70/30 split.

There are several options for specifying the model architecture, communication
protocol, dataset, and testing (or not testing) specific aspects of
generalization. This is all fairly well documented via `python rnn-syn.py
--help`.

## Analysis

`rnn-syn.ipynb` contains analysis and visualization of the communication
protocol. It will also (hopefully) contain procedures for generating new
messages and evaluating their accuracies.
