# Compositional communication in end-to-end, multi-agent reference games

To generate a spatial relations dataset:

```bash
python rnn-syn.py --no_train --n_configs 15 --samples_each_config 100 \
    --n_targets 2 --n_distractors 1 --n_cpu 1
```

creates a gzipped pickle in
`data/{num_scenes}-{num_configs}-{samples_each_config}-{n_targets}t-{n_distractors}d.pkl.gz` by default.

(the above numbers are the default values of the relevant options).

then run

```pythonbash
python rnn-syn.py --load_dataset data/365-15-100-2t-1d.pkl.gz
```

You can also generate datasets on-the-fly in `rnn-syn` by omitting
`--load_dataset`, but this takes longer.
