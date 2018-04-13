"""
Generate specific image components which are them sampled from in rnn-syn.py.
"""

from swdata import parse_configs, SpatialExtraSimple
import itertools
from scipy.misc import imsave
import numpy as np
import pickle
import gzip
import os
import multiprocessing as mp


def dbg_save(img, name='test.png'):
    imsave(name, (img * 255).astype(np.uint8))


def config_str(tuple):
    return '{}-{}'.format(tuple[0], tuple[1])


def gen_components(cc_both, cc_rel, n, new_only=False):
    cc_both = sorted(cc_both)
    cc_target, cc_distractor = cc_both
    cc_fname = os.path.join(
        './data/components/',
        '{}-{}-{}-{}-{{}}.pkl.gz'.format(
            n, config_str(cc_target),
            config_str(cc_distractor), cc_rel[0])
    )
    t_fname = cc_fname.format('t')
    d_fname = cc_fname.format('d')

    if new_only and os.path.exists(t_fname):
        # Assume d_fname exists too
        print("Skipping {} {} as it already exists".format(cc_both, cc_rel))
        return

    print("Starting", cc_both, cc_rel)
    dataset = SpatialExtraSimple(target=cc_target,
                                 distractor=cc_distractor,
                                 relation=cc_rel,
                                 # Always relation 1 - can sample pos/neg
                                 # examples later
                                 relation_dir=1)
    targets, distractors = dataset.generate_targets_distractors(n * 2)

    for fname, comps in [(t_fname, targets), (d_fname, distractors)]:
        with gzip.open(fname, 'w') as fout:
            pickle.dump(comps, fout, protocol=pickle.HIGHEST_PROTOCOL)
    print("Done", cc_both, cc_rel)


if __name__ == '__main__':
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

    parser = ArgumentParser(
        description='Generate specific images',
        formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        '--configs', type=str,
        # 20000 each here
        default='square-blue,square-red,triangle-blue,triangle-red,'
                'triangle-green,square-green',
        # Add a triangle-green to the mix
        #  default='square-blue,square-red,triangle-blue,triangle-green',
        help='Manually specify possible configs '
             'as `color-shape` pairs, comma-separated')
    parser.add_argument('--new_only', action='store_true',
                        help='Only generate configs that don\'t yet exist')
    parser.add_argument('--n_per_config', type=int,
                        default=20000,
                        help='Approx number of targets/distractors per '
                             'config')
    parser.add_argument('--n_cpu', type=int,
                        default=6)

    args = parser.parse_args()

    configs = parse_configs(args.configs)

    config_combs = list(itertools.combinations(configs, 2))
    relations = ('x-rel', 'y-rel')
    config_combs = list(itertools.product(config_combs, relations))

    print(
        "Need to generate targets/distractors for {} configs".format(
            len(config_combs)))

    mp_args = [t + (args.n_per_config, args.new_only)
               for t in config_combs]

    pool = mp.Pool(args.n_cpu)
    pool.starmap(gen_components, mp_args)
    pool.close()
    pool.join()
