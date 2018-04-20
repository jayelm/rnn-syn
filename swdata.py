"""
Generate random shapeworld data
"""

from shapeworld.dataset import CaptionAgreementDataset
from shapeworld.generators import RandomAttributesGenerator
from shapeworld.captioners import ExistentialCaptioner, RegularTypeCaptioner, RelationCaptioner
from shapeworld.world import World

import numpy as np
from PIL import Image
from tqdm import tqdm

import os
import sys
import shutil
from glob import glob
import json
from collections import namedtuple
import pickle
import gzip
from itertools import cycle
import time
import multiprocessing as mp
import subprocess
from random import choice as choice1d
from collections import defaultdict


random = np.random.RandomState()


def weighted_choice(choices):
    total = sum(w for c, w in choices)
    r = random.uniform(0, total)
    upto = 0
    for c, w in choices:
        if upto + w >= r:
            return c
        upto += w
    assert False, "Shouldn't get here"


def invert(d):
    return {v: k for k, v in d.items()}


def sample_asym(max_images, min_targets, min_distractors):
    """
    Sample a random number of targets between min_targets and
    max_images - min_distractors
    """
    n_targets = random.randint(
        min_targets, max_images - min_distractors + 1)
    n_left = max_images - n_targets
    assert n_left >= min_distractors
    # Sample a number of distractors between min_distractors and
    # the number of possible images left
    n_distractors = random.randint(
        min_distractors, n_left + 1)
    assert (n_targets + n_distractors) <= max_images
    return n_targets, n_distractors


COLORS = [
    'red',
    'green',
    'blue',
    'yellow',
    'magenta',
    'cyan',
]
COLORS_MAP = invert(dict(enumerate(COLORS)))

NAMES = [
    'ellipse', 'pentagon', 'rectangle', 'semicircle', 'square', 'cross',
    'circle', 'triangle'
]
NAMES_MAP = invert(dict(enumerate(NAMES)))

TEXTURES = ['solid']
TEXTURES_MAP = invert(dict(enumerate(TEXTURES)))

Scene = namedtuple('Scene', ['worlds', 'labels', 'relation', 'relation_dir'])
AsymScene = namedtuple('AsymScene', ['speaker_worlds', 'speaker_labels',
                                     'listener_worlds', 'listener_labels',
                                     'relation', 'relation_dir'])
SWorld = namedtuple('SWorld', ['image', 'shapes'])
Shape = namedtuple('Shape',
                   ['name', 'color', 'size', 'center', 'rotation', 'texture'])
Point = namedtuple('Point', ['x', 'y'])
Color = namedtuple('Color', ['color', 'shade'])

RELATIONS = ['x-rel', 'y-rel']


VOCABULARY = sorted([
    '.', 'a', 'above', 'an', 'below', 'blue', 'circle', 'cross',
    'cyan', 'ellipse', 'gray', 'green', 'is', 'left', 'magenta', 'of',
    'pentagon', 'rectangle', 'red', 'right', 'semicircle', 'shape',
    'square', 'the', 'to', 'triangle', 'yellow'
])


# Used for training
TrainEx = namedtuple('TrainEx', ['world', 'metadata'])


# Parse an argparse --configs string
def parse_configs(configs_str):
    configs = [tuple(c.split('-')) + ('solid', )
               for c in configs_str.split(',')]
    if not all(len(c) == 3 for c in configs):
        raise ValueError("Invalid config format")
    return configs


def pickle_scenes(scenes, save_file='data/dataset.pkl', gz=True):
    if save_file == 'data/dataset.pkl' and gz:
        save_file += '.gz'

    dirs, fname = os.path.split(save_file)
    os.makedirs(dirs, exist_ok=True)

    opener = gzip.open if gz else open
    with opener(save_file, 'wb') as fout:
        pickle.dump(scenes, fout, protocol=pickle.HIGHEST_PROTOCOL)


def load_scenes(folder, gz=True):
    metadata_file = os.path.join(folder, 'metadata.json')
    with open(metadata_file, 'r') as fin:
        metadata = json.load(fin)

    opener = gzip.open if gz else open
    glob_pattern = '*.pkl.gz' if gz else '*.pkl'
    global_scenes = []
    for scene_file in glob(os.path.join(folder, glob_pattern)):
        with opener(scene_file, 'rb') as fin:
            try:
                scenes = pickle.load(fin)
            except AttributeError:
                raise RuntimeError(
                    "Can't find Scene/SWorld.\n"
                    "   include `from swdata import AsymScene, Scene, SWorld`")
            global_scenes.append(scenes)
    return global_scenes, metadata


def flatten(l, with_metadata=False):
    if with_metadata:
        flat = []
        for sublist, md in l:
            for item in sublist:
                flat.append(TrainEx(item, md))
        return flat
    return [item for sublist in l for item in sublist]


def train_test_split(train, test_split):
    test_i = int(len(train) * test_split)
    return train[test_i:], train[:test_i]


def max_scenes(len_targets, len_distractors, n_targets, n_distractors):
    """
    How many scenes can be constructed with the given distractor/target
    count?
    """
    max_target_scenes = len_targets // n_targets
    max_distractor_scenes = len_distractors // n_distractors
    return min(max_target_scenes, max_distractor_scenes)


def comp_obj(entity_model, obj):
    # e.g. (square, red, solid)
    for i, prop in enumerate(['shape', 'color', 'texture']):
        if prop in entity_model and entity_model[prop]['value'] != obj[i]:
            return False
    return True


def flatten_scene(scene):
    """Flatten a to make it tf-compatible"""
    new_worlds = [
        SWorld(image=s.image,
               shapes=flatten_shapes(s.shapes))
        for s in scene.worlds
    ]
    return Scene(worlds=new_worlds, labels=scene.labels,
                 relation=scene.relation,
                 relation_dir=scene.relation_dir)


def flatten_asym_scene(scene):
    """Flatten an asym scene to make it tf-compatible"""
    new_speaker_worlds = [
        SWorld(image=s.image,
               shapes=flatten_shapes(s.shapes))
        for s in scene.speaker_worlds
    ]
    new_listener_worlds = [
        SWorld(image=s.image,
               shapes=flatten_shapes(s.shapes))
        for s in scene.listener_worlds
    ]
    return AsymScene(speaker_worlds=new_speaker_worlds,
                     speaker_labels=scene.speaker_labels,
                     listener_worlds=new_listener_worlds,
                     listener_labels=scene.listener_labels,
                     relation=scene.relation,
                     relation_dir=scene.relation_dir)


def flatten_shapes(shapes):
    return np.array(list(map(flatten_shape, shapes)),
                    dtype=np.float32)


def flatten_shape(shape):
    """Return structured array representation of shape"""
    names_onehot = [0 for _ in NAMES]
    names_onehot[NAMES_MAP[shape.name]] = 1
    colors_onehot = [0 for _ in COLORS]
    colors_onehot[COLORS_MAP[shape.color.color]] = 1
    textures_onehot = [0 for _ in TEXTURES]
    textures_onehot[TEXTURES_MAP[shape.texture]] = 1
    reals = [
        shape.color.shade, shape.size.x, shape.size.y, shape.center.x,
        shape.center.y, shape.rotation
    ]
    return np.array(names_onehot + colors_onehot + textures_onehot + reals,
                    dtype=np.float32)


def extract_envs_and_labels(scenes, n_images, max_shapes, n_attrs, asym=False,
                            shuffle=True):
    """
    Given a list of scenes, return a list of tf-compatible feature reps and
    labels
    """
    if asym:
        speaker_envs, speaker_labels = _features_from_scenes(
            scenes, n_images, max_shapes, n_attrs,
            props=['speaker_worlds', 'speaker_labels'])
        listener_envs, listener_labels = _features_from_scenes(
            scenes, n_images, max_shapes, n_attrs,
            props=['listener_worlds', 'listener_labels'])
        if shuffle:
            speaker_envs, speaker_labels = shuffle_envs_labels(
                speaker_envs, speaker_labels)
            listener_envs, listener_labels = shuffle_envs_labels(
                listener_envs, listener_labels)
        return speaker_envs, speaker_labels, listener_envs, listener_labels
    else:
        envs, labels = _features_from_scenes(
            scenes, n_images, max_shapes, n_attrs,
            props=['worlds', 'labels'])
        if shuffle:
            envs, labels = shuffle_envs_labels(envs, labels)
        return envs, labels



def _features_from_scenes(scenes, n_images, max_shapes,
                          n_attrs, props=('worlds', 'labels')):
    """
    Extract envs from a list of scenes, but can modify props for AsymScenes
    """
    n_scenes = len(scenes)
    envs = np.zeros((n_scenes, n_images, max_shapes * n_attrs))
    labels = np.zeros((n_scenes, n_images))

    for scene_i, scene in enumerate(scenes):
        worlds_prop = getattr(scene, props[0])
        labels_prop = getattr(scene, props[1])
        for world_i, wl in enumerate(zip(worlds_prop, labels_prop)):
            world, label = wl
            global_shape_i = 0
            for shape in world.shapes:
                for shape_prop in shape:
                    envs[scene_i, world_i, global_shape_i] = shape_prop
                    global_shape_i += 1
            labels[scene_i, world_i] = label

    return envs, labels


def _end2end_from_scenes(scenes, n_images, props=('worlds', 'labels')):
    """
    Extract images from a list of scenes, but can modify props for AsymScenes
    """
    n_scenes = len(scenes)

    image_dim = getattr(scenes[0], props[0])[0].image.shape

    envs = np.zeros((n_scenes, n_images) + image_dim)
    labels = np.zeros((n_scenes, n_images))

    for scene_i, scene in enumerate(scenes):
        worlds_prop = getattr(scene, props[0])
        labels_prop = getattr(scene, props[1])
        for img_i, wl in enumerate(zip(worlds_prop, labels_prop)):
            world, label = wl
            envs[scene_i, img_i, :] = world.image
            labels[scene_i, img_i] = label

    return envs, labels


def shuffle_envs_labels(envs, labels):
    new_envs = np.zeros_like(envs)
    new_labels = np.zeros_like(labels)
    world_seq = list(range(envs[0].shape[0]))
    # Loop through each world (env/label) in the batch
    for env_i, (env, label) in enumerate(zip(envs, labels)):
        # New sequence of worlds to retrieve from original envs/labels
        random.shuffle(world_seq)
        # Loop through new sequence, place this sequence in increasing order
        # in new_envs/labels
        for new_world_i, orig_world_i in enumerate(world_seq):
            new_envs[env_i, new_world_i] = env[orig_world_i]
            new_labels[env_i, new_world_i] = label[orig_world_i]
    return new_envs, new_labels


def prepare_end2end(scenes, n_images, asym=False, shuffle=True):
    """
    Given a list of scenes, return a list of tf-compatible feature reps and
    labels
    """
    if asym:
        speaker_envs, speaker_labels = _end2end_from_scenes(
            scenes, n_images,
            props=['speaker_worlds', 'speaker_labels'])
        listener_envs, listener_labels = _end2end_from_scenes(
            scenes, n_images,
            props=['listener_worlds', 'listener_labels'])
        if shuffle:
            speaker_envs, speaker_labels = shuffle_envs_labels(
                speaker_envs, speaker_labels)
            listener_envs, listener_labels = shuffle_envs_labels(
                listener_envs, listener_labels)
        return speaker_envs, speaker_labels, listener_envs, listener_labels
    else:
        envs, labels = _end2end_from_scenes(
            scenes, n_images,
            props=['worlds', 'labels'])
        if shuffle:
            envs, labels = shuffle_envs_labels(envs, labels)
        return envs, labels


class FixedWorldGenerator(RandomAttributesGenerator):
    def generate_validation_world(self):
        world = World(self.world_size, self.world_color)
        if self.num_entities == 0:
            return world
        n = 0
        last_entity = -1
        assert self.num_entities <= len(self.validation_combinations), \
            "insufficient validation combs"
        if self.validation_combinations:
            remain_combs = list(self.validation_combinations)
            while True:
                entity = self.sample_entity(
                    world=world,
                    last_entity=last_entity,
                    combinations=[remain_combs[n]])
                if world.add_entity(
                        entity,
                        boundary_tolerance=self.boundary_tolerance,
                        collision_tolerance=self.collision_tolerance):
                    last_entity = entity
                    n += 1
                    break
                else:
                    last_entity = None
        if n < self.num_entities:
            for _ in range(self.num_entities * self.__class__.MAX_ATTEMPTS):
                entity = self.sample_entity(
                    world=world,
                    last_entity=last_entity,
                    combinations=[remain_combs[n]])
                if world.add_entity(
                        entity,
                        boundary_tolerance=self.boundary_tolerance,
                        collision_tolerance=self.collision_tolerance):
                    last_entity = entity
                    n += 1
                    if n == self.num_entities:
                        break
                else:
                    last_entity = None
            else:
                return None
        if self.collision_tolerance:
            world.sort_entities()
        return world


def random_objects(n):
    objs = []
    for _ in range(n):
        objs.append((random.choice(NAMES), random.choice(COLORS),
                     random.choice(TEXTURES)))
    return objs


def sw_arr_to_img(swarr):
    return Image.fromarray(np.uint8(swarr * 255))


def to_sworld(sw):
    """
    """
    img, world_model = sw
    shapes = get_shapes(world_model)
    return SWorld(image=img, shapes=shapes)


def get_shapes(world_model):
    shapes = []
    for shape_model in world_model['entities']:
        shape = Shape(
            name=shape_model['shape']['name'],
            color=Color(color=shape_model['color']['name'],
                        shade=shape_model['color']['shade']),
            size=Point(**shape_model['shape']['size']),
            center=Point(**shape_model['center']),
            rotation=shape_model['rotation'],
            texture=shape_model['texture']['name']
        )
        shapes.append(shape)
    return shapes


class SpatialExtraSimple(CaptionAgreementDataset):
    """
    A dataset that generates target/distractor pairs with common objects and
    simple relations.
    """

    def __init__(self,
                 target=None,
                 distractor=None,
                 combinations=None,
                 relation=None,
                 relation_dir=None):
        # Randomly sample combinations and relations if not provided
        if (target is not None and
                distractor is not None and combinations):
            raise ValueError(
                "Can't specify both target/distractor and possible "
                "combinations")

        if isinstance(target, list):
            target_combinations = choice1d(target)
        elif isinstance(target, tuple):
            target_combinations = target
        elif target is None:
            if isinstance(combinations, list):
                if len(combinations) < 1:
                    raise ValueError("Need more combinations")
                target_combinations = choice1d(combinations)
                assert isinstance(target_combinations, tuple)
            elif combinations is None:
                target_combinations, = random_objects(1)
            else:
                raise ValueError
        else:
            raise ValueError

        if isinstance(distractor, list):
            distractor_combinations = random.choice(distractor)
        elif isinstance(distractor, tuple):
            distractor_combinations = distractor
        elif distractor is None:
            distractor_combinations = (None, None)
            while (distractor_combinations == (None, None) or
                    distractor_combinations == target_combinations):
                # TODO: Support multiple distractors
                if isinstance(combinations, list):
                    if len(combinations) < 1:
                        raise ValueError("Need more combinations")
                    distractor_combinations = choice1d(combinations)
                    assert isinstance(distractor_combinations, tuple)
                elif combinations is None:
                    distractor_combinations, = random_objects(1)
                else:
                    raise ValueError
        else:
            raise ValueError

        self.target_i = random.randint(2)
        self.target_obj = target_combinations
        # TODO: Make this distractor_objs, supportm multiple distractors
        self.distractor_obj = distractor_combinations

        if self.target_i == 0:
            self.shapes = [target_combinations, distractor_combinations]
        else:
            self.shapes = [distractor_combinations, target_combinations]

        # Relations and directories
        if relation is None:
            relation = random.choice(RELATIONS)
        assert relation in ('x-rel', 'y-rel'), "Invalid relation"
        self.relation = relation

        if relation_dir is None:
            self.relation_dir = random.choice([1, -1])
        else:
            self.relation_dir = relation_dir

        vocabulary = VOCABULARY

        world_generator = FixedWorldGenerator(
            entity_counts=[2],
            validation_combinations=tuple(self.shapes[:]),
            test_combinations=tuple(self.shapes[:]),
            max_provoke_collision_rate=0.0,
            collision_tolerance=0.0,
            boundary_tolerance=0.0)

        world_captioner = ExistentialCaptioner(
            restrictor_captioner=RegularTypeCaptioner(),
            body_captioner=RelationCaptioner(
                reference_captioner=RegularTypeCaptioner(),
                comparison_captioner=RegularTypeCaptioner(),
                relations=(relation, )),
            pragmatical_tautology_rate=1.0)

        super().__init__(
            world_generator=world_generator,
            world_captioner=world_captioner,
            caption_size=12,
            vocabulary=vocabulary,
            language=None,
            correct_ratio=1.00,
        )

    def generate_shapeworld(self, max, noise_range=0.0):
        return super().generate(
            max,
            mode='validation',
            noise_range=noise_range,
            include_model=True)

    @property
    def relation_str(self):
        if self.relation_dir == 1 and self.relation == 'y-rel':
            return 'below'
        elif self.relation_dir == -1 and self.relation == 'y-rel':
            return 'above'
        elif self.relation_dir == 1 and self.relation == 'x-rel':
            return 'right'
        elif self.relation_dir == -1 and self.relation == 'x-rel':
            return 'left'

    def generate_targets_distractors(self, max, noise_range=0.0):
        sw_data = self.generate_shapeworld(max, noise_range=noise_range)
        targets = []
        distractors = []

        for img, wm, cm in zip(sw_data['world'],
                               sw_data['world_model'],
                               sw_data['caption_model']):

            cap_target = cm['restrictor']['value']
            cap_relation_dir = cm['body']['value']
            combs_is_target = [comp_obj(cap_target, c) for c in self.shapes]
            if sum(combs_is_target) == 0:
                raise RuntimeError("No compatible referent! {} ({})".format(
                    cm, self.shapes))
            elif sum(combs_is_target) > 1:
                # Ambiguous referent
                opp_target = cm['body']['reference']['value']
                opps_is_target = [comp_obj(opp_target, c) for c in self.shapes]
                if sum(opps_is_target) == 0:
                    raise RuntimeError("No opposite referent! {} ({})".format(
                        cm, self.shapes))
                elif sum(opps_is_target) > 1:
                    continue  # Both targets ambiguous
                cap_target_i = opps_is_target.index(False)
            else:
                cap_target_i = combs_is_target.index(True)

            combined = ((img, wm))
            if self.relation_dir == cap_relation_dir:
                if self.target_i == cap_target_i:
                    targets.append(combined)
                else:
                    distractors.append(combined)
            else:
                # Since the relation direction is flipped,
                # the targets are flipped
                if self.target_i == cap_target_i:
                    distractors.append(combined)
                else:
                    targets.append(combined)

        return targets, distractors

    def generate(self, max, noise_range=0.0,
                 n_targets=2,
                 n_distractors=1,
                 asym=False, asym_args=None):
        if asym_args is None:
            asym_args = {
                'max_images': 5,
                'min_targets': 2,
                'min_distractors': 1
            }
        targets, distractors = self.generate_targets_distractors(
            max, noise_range=noise_range)
        if asym:
            return self.combine_targets_distractors_asym(
                targets, distractors, **asym_args)
        else:
            return self.combine_targets_distractors(
                targets, distractors,
                n_targets=n_targets, n_distractors=n_distractors)

    def combine_targets_distractors_asym(self, targets, distractors,
                                         max_images=5,
                                         min_targets=2,
                                         min_distractors=1):
        if min_targets + min_distractors > max_images:
            raise ValueError("Min targets + min distractors > max images")
        if min_targets + min_distractors == max_images:
            print("Warning: min_targets + min_distractors == max_images, "
                  "will always produce scenes with `max_images` images")
        asym_scenes = []
        try:
            while True:
                n_targets_speaker, n_distractors_speaker = sample_asym(
                    max_images, min_targets, min_distractors)
                n_targets_listener, n_distractors_listener = sample_asym(
                    max_images, min_targets, min_distractors)
                speaker_targets = [(targets.pop(), 1)
                                   for _ in range(n_targets_speaker)]
                speaker_distractors = [(distractors.pop(), 0)
                                       for _ in range(n_distractors_speaker)]
                listener_targets = [(targets.pop(), 1)
                                    for _ in range(n_targets_listener)]
                listener_distractors = [(distractors.pop(), 0)
                                        for _ in range(n_distractors_listener)]
                speaker_combs = speaker_targets + speaker_distractors
                listener_combs = listener_targets + listener_distractors
                random.shuffle(speaker_combs)
                random.shuffle(listener_combs)
                speaker_sworlds, speaker_labels = zip(*speaker_combs)
                listener_sworlds, listener_labels = zip(*listener_combs)
                speaker_sworlds = list(map(to_sworld, speaker_sworlds))
                listener_sworlds = list(map(to_sworld, listener_sworlds))
                scene = AsymScene(speaker_worlds=speaker_sworlds,
                                  speaker_labels=np.array(speaker_labels),
                                  listener_worlds=listener_sworlds,
                                  listener_labels=np.array(listener_labels),
                                  relation=self.relation,
                                  relation_dir=self.relation_dir)
                scene = flatten_asym_scene(scene)
                asym_scenes.append(scene)
        except IndexError:
            assert not targets or not distractors
            return asym_scenes

    def combine_targets_distractors(self,
                                    targets,
                                    distractors,
                                    n_targets=2,
                                    n_distractors=1):
        n_scenes = max_scenes(len(targets), len(distractors),
                              n_targets, n_distractors)
        if n_scenes == 0:
            raise ValueError("Not enough targets/distractors for 1 scene")
        if n_scenes == 1:
            print("Warning: only making 1 scene")

        scenes = []
        for scene_i in range(n_scenes):
            scene_targets = [(targets.pop(), 1)
                             for _ in range(n_targets)]
            scene_distractors = [(distractors.pop(), 0)
                                 for _ in range(n_distractors)]

            combs = scene_targets + scene_distractors
            random.shuffle(combs)
            sworlds, labels = zip(*combs)
            sworlds = list(map(to_sworld, sworlds))

            scene = Scene(worlds=sworlds, labels=np.array(labels),
                          relation=self.relation,
                          relation_dir=self.relation_dir)
            scene = flatten_scene(scene)
            scenes.append(scene)

        return scenes

    def to_html(self, scenes, save_dir='test'):
        """
        Save images as bmps and html index to save_dir
        """
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        html_str = """
            <!DOCTYPE html>
            <html>
                <head>
                    <title>temp</title>
                    <style>
                    img {
                        display: block;
                    }
                    .scene {
                        display: block;
                        margin-top: 20px;
                        margin-bottom: 20px;
                    }
                    .target {
                        display: inline-block;
                        padding: 10px;
                        background-color: #99ff99;
                    }
                    .distractor {
                        display: inline-block;
                        padding: 10px;
                        background-color: #ff9999;
                    }
                    </style>
                </head>
            <body>
        """
        target_obj_str = ' '.join(reversed(self.target_obj))
        opp_obj_str = ' '.join(reversed(self.shapes[1 - self.target_i]))
        if self.relation == 'y-rel':
            relation_str_nice = self.relation_str
        else:
            relation_str_nice = 'to the {} of'.format(self.relation_str)
        html_str += "<p>Target: {}<br>Relation: {}<br>Direction: {}<br>Caption: {}</p>".format(
            target_obj_str, self.relation,
            self.relation_dir, 'A {} is {} a {}'.format(
                target_obj_str, relation_str_nice, opp_obj_str))

        global_i = 0
        for i, scene in enumerate(scenes):
            html_str += "<div class='scene' id='{}'>".format(i)

            for sw_i, swlabel in enumerate(zip(scene.worlds, scene.labels)):
                sw, label = swlabel
                label_str = 'target' if label else 'distractor'
                # Save image
                img = sw_arr_to_img(sw.image)
                world_name = 'world-{}-{}.bmp'.format(global_i, label_str)
                global_i += 1
                img.save(os.path.join(save_dir, world_name))

                # Add link to html with given class
                html_str += """
                    <div class='{}'><img src='{}'></img></div>
                """.format(label_str, world_name)
            html_str += "</div>"

        html_str += "</body></html>"

        with open(os.path.join(save_dir, 'index.html'), 'w') as fout:
            fout.write(html_str)
            fout.write('\n')

    @staticmethod
    def to_html_many(data, save_dir='test'):
        """
        Save images as bmps and html index to save_dir,
        without captions, etc
        """
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        html_str = """
            <!DOCTYPE html>
            <html>
                <head>
                    <title>temp</title>
                    <style>
                    img {
                        display: block;
                    }
                    .scene {
                        display: block;
                        margin-top: 20px;
                        margin-bottom: 20px;
                    }
                    .target {
                        display: inline-block;
                        padding: 10px;
                        background-color: #99ff99;
                    }
                    .distractor {
                        display: inline-block;
                        padding: 10px;
                        background-color: #ff9999;
                    }
                    </style>
                </head>
            <body>
        """
        global_i = 0
        for i, scene in enumerate(data):
            html_str += "<div class='scene' id='{}'>".format(i)

            for sw_i, swlabel in enumerate(zip(scene.worlds, scene.labels)):
                sw, label = swlabel
                label_str = 'target' if label else 'distractor'
                # Save image
                img = sw_arr_to_img(sw.image)
                world_name = 'world-{}-{}.bmp'.format(global_i, label_str)
                global_i += 1
                img.save(os.path.join(save_dir, world_name))

                # Add link to html with given class
                html_str += """
                    <div class='{}'><img src='{}'></img></div>
                """.format(label_str, world_name)
            html_str += "</div>"

        html_str += "</body></html>"

        with open(os.path.join(save_dir, 'index.html'), 'w') as fout:
            fout.write(html_str)
            fout.write('\n')


def load_components(component_strs, component_path='./data/components/',
                    maxdata=256, n_cpu=1):
    components_dict = defaultdict(dict)
    configs = []
    for cstr in tqdm(component_strs, desc='Load components'):
        if cstr.endswith('-x') or cstr.endswith('-y'):
            raise NotImplementedError("Can't test specific relations")
        cstr_path = os.path.join(component_path, cstr)

        # Extract target and distractor by shaving off the number
        cstr_td_pieces = cstr.split('-')[1:]
        target = tuple(cstr_td_pieces[:2])
        distractor = tuple(cstr_td_pieces[2:])

        for rel in ['x', 'y']:
            config_hash = (target, distractor, '{}-rel'.format(rel))
            assert config_hash not in configs
            configs.append(config_hash)
            for td in ['targets', 'distractors']:
                # Load either targets or distractors
                cstr_td = '{}-{}-{}.pkl.gz'.format(cstr_path, rel, td[0])
                with gzip.open(cstr_td, 'r') as f_components:
                    components_dict[config_hash][td] = pickle.load(
                        f_components)[:maxdata]
    return configs, dict(components_dict)


def make_from_components(n, configs, components_dict, asym=True,
                         asym_args=None,
                         relation_dir=None, weighted=False):
    """
    Sample `n` training examples
    """
    max_images = asym_args['max_images']
    min_targets = asym_args['min_targets']
    min_distractors = asym_args['min_distractors']
    scenes = []
    for i in range(n):
        if weighted:
            config = weighted_choice(configs)
        else:
            config = choice1d(configs)
        target, distractor, rel = config
        if relation_dir is None:
            # Sample a new rd
            rd = 1 if random.randint(2) else -1
        elif isinstance(relation_dir, int):
            assert relation_dir == 1 or relation_dir == -1
            rd = relation_dir
        else:
            raise ValueError("Unknown relation dir {}".format(relation_dir))
        cc_targets = components_dict[config]['targets']
        cc_distractors = components_dict[config]['distractors']
        n_targets_speaker, n_distractors_speaker = sample_asym(
            max_images, min_targets, min_distractors)
        n_targets_listener, n_distractors_listener = sample_asym(
            max_images, min_targets, min_distractors)
        targets_yes = 1 if rd == 1 else 0
        distractors_yes = 0 if rd == 1 else 1
        speaker_targets = [(choice1d(cc_targets), targets_yes)
                           for _ in range(n_targets_speaker)]
        speaker_distractors = [(choice1d(cc_distractors), distractors_yes)
                               for _ in range(n_distractors_speaker)]
        if asym:
            listener_targets = [(choice1d(cc_targets), targets_yes)
                                for _ in range(n_targets_listener)]
            listener_distractors = [(choice1d(cc_distractors), distractors_yes)
                                    for _ in range(n_distractors_listener)]
        else:
            listener_targets = speaker_targets[:]
            listener_distractors = speaker_distractors[:]
        speaker_combs = speaker_targets + speaker_distractors
        listener_combs = listener_targets + listener_distractors
        random.shuffle(speaker_combs)
        random.shuffle(listener_combs)
        speaker_sworlds, speaker_labels = zip(*speaker_combs)
        listener_sworlds, listener_labels = zip(*listener_combs)
        speaker_sworlds = list(map(to_sworld, speaker_sworlds))
        listener_sworlds = list(map(to_sworld, listener_sworlds))
        scene = AsymScene(speaker_worlds=speaker_sworlds,
                          speaker_labels=np.array(speaker_labels),
                          listener_worlds=listener_sworlds,
                          listener_labels=np.array(listener_labels),
                          relation=rel,
                          relation_dir=rd)
        scene = flatten_asym_scene(scene)
        metadata = {
            'config': i,
            'n': 1,
            'relation': config[-1],
            'relation_dir': rd,
            'target': list(target) + ['solid'],
            'distractor': list(distractor) + ['solid'],
        }
        scenes.append(TrainEx(scene, metadata))
    return scenes


def gen_dataset(iargs):
    i, args = iargs
    if i is not None:
        t = time.time()
        print("{} Started".format(i))
        sys.stdout.flush()
    (n, max_n, n_targets, n_distractors,
     save_folder, asym, asym_args, target, distractor,
     configs, pickle) = args
    dataset = SpatialExtraSimple(combinations=configs)
    train = dataset.generate(
        max_n, n_targets=n_targets, n_distractors=n_distractors,
        asym=asym, asym_args=asym_args)
    if i is not None:
        print("{} Finished ({}s)".format(i, round(time.time() - t, 2)))
        sys.stdout.flush()
    metadata = {
        'config': n,
        'n': len(train),
        'relation': dataset.relation,
        'relation_dir': int(dataset.relation_dir),
        'target': dataset.target_obj,
        'distractor': dataset.distractor_obj
    }
    # Save data
    if pickle:
        save_file = '{}.pkl.gz'.format(n)
        pickle_scenes(
            train, save_file=os.path.join(save_folder, save_file), gz=True)
        return metadata
    # Otherwise, return the dataset itself with the metadata
    return train, metadata


def gen_datasets(n_configs, samples_each_config,
                 n_targets, n_distractors, target=None, distractor=None,
                 configs=None, save_folder=None,
                 asym=False, asym_args=None, pickle=False,
                 n_cpu=1):
    if pickle and save_folder is None:
        raise ValueError("Must specify save_folder if pickling")

    args1 = (samples_each_config, n_targets,
             n_distractors, save_folder,
             asym, asym_args, target, distractor, configs,
             pickle)
    # Index the datasets by config
    dataset_args = [(i, ) + args1 for i in range(n_configs)]

    if n_cpu == 1:  # Non-mp, track progress with tqdm
        dataset_metas = []
        if not pickle:
            trains = []
        for dargs in tqdm(list(zip(cycle([None]), dataset_args))):
            if pickle:
                dataset_meta = gen_dataset(dargs)
                dataset_metas.append(dataset_meta)
            else:
                train, dataset_meta = gen_dataset(dargs)
                dataset_metas.append(dataset_meta)
                trains.append(train)
    else:
        t = time.time()
        print("Multiprocessing")
        pool = mp.Pool(n_cpu)
        if pickle:
            dataset_metas = pool.map(gen_dataset, list(enumerate(dataset_args)))
        else:
            trains, dataset_metas = zip(
                *pool.map(gen_dataset, list(enumerate(dataset_args))))
        pool.close()
        pool.join()
        print("Elapsed time: {}s".format(round(time.time() - t, 2)))

    metadata = {
        'n': sum(d['n'] for d in dataset_metas),
        'n_configs': n_configs,
        'samples_each_config': samples_each_config,
        'n_targets': None if asym else n_targets,
        'n_distractors': None if asym else n_distractors,
        'asym': True,
        'asym_args': None if not asym else asym_args,
        'configs': dataset_metas
    }

    if pickle:
        # Save dataset metadata and return nothing
        print("Writing metadata")
        with open(os.path.join(save_folder, 'metadata.json'), 'w') as fout:
            json.dump(metadata, fout, sort_keys=True,
                      indent=2, separators=(',', ': '))
        print("Done")
    else:
        # Return training data and dataset metas
        return trains, metadata


if __name__ == "__main__":
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter,
        description='Generate spatial datasets')
    parser.add_argument(
        '--n_configs',
        type=int,
        default=15,
        help='Number of random scene configs to sample')
    parser.add_argument(
        '--samples_each_config',
        type=int,
        default=100,
        help='(max) number of scenes to sample per config')
    parser.add_argument(
        '--n_targets', type=int, default=2, help='Number of targets per scene')
    parser.add_argument(
        '--n_distractors',
        type=int,
        default=1,
        help='Number of distractors per scene')
    parser.add_argument(
        '--asym',
        action='store_true',
        help='Construct scenes with different (random) images on '
             'speaker/listener side')
    parser.add_argument('--configs', type=str, default='',
                        help='Manually specify possible configs '
                             'as `color-shape` pairs, comma-separated')
    parser.add_argument(
        '--save_folder',
        default='data/{configs}{n_configs}_{samples_each_config}_asym{asym}',
        help='Save folder (can use other args)')
    parser.add_argument(
        '--n_cpu',
        type=int,
        default=1,
        help='Number of cpus to use for mp (1 disables mp)')
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite data folder if already exists')
    parser.add_argument(
        '--verify_load',
        action='store_true',
        help='Verify data by reloading it')
    parser.add_argument(
        '--compress',
        action='store_true',
        help='Compress (tar.gz) data after generating')

    asym_args = parser.add_argument_group('Asym args',
                                          'options for asymmetric worlds')
    asym_args.add_argument(
        '--asym_max_images', default=5, type=int,
        help='Maximum images in each asymmetric world')
    asym_args.add_argument(
        '--asym_min_targets', default=2, type=int,
        help='Minimum targets in each asymmetric world')
    asym_args.add_argument(
        '--asym_min_distractors', default=1, type=int,
        help='Minimum distractors in each asymmetric world')

    args = parser.parse_args()

    if args.configs:
        configs = parse_configs(args.configs)
        # Make configs a slightly better format
        args.configs = '_'.join('-'.join(c) for c in sorted(configs))
    else:
        configs = None

    # Check save folder.
    save_folder = args.save_folder.format(**vars(args))
    if os.path.exists(save_folder):
        overwrite_msg = (
            'Overwrite {} and .tar.gz, if exists? [y/N] '.format(save_folder)
        )
        if args.overwrite or input(overwrite_msg).lower().startswith('y'):
            shutil.rmtree(save_folder)
            poss_targz = save_folder + '.tar.gz'
            if os.path.exists(poss_targz):
                os.remove(poss_targz)
        else:
            print("Exiting")
            sys.exit(0)
    os.mkdir(save_folder)

    asym_args = {
        'max_images': args.asym_max_images,
        'min_targets': args.asym_min_targets,
        'min_distractors': args.asym_min_distractors,
    }

    print("Generating data")
    gen_datasets(
        args.n_configs, args.samples_each_config,
        args.n_targets, args.n_distractors, configs=configs,
        save_folder=save_folder,
        asym=args.asym, asym_args=asym_args, pickle=True)

    if args.verify_load:
        try:
            _ = load_scenes(save_folder)
        except Exception as e:
            print("\nDataset verification failed:\n")
            raise

    if args.compress:
        print("Compressing")
        up_to_sf, sf = os.path.split(os.path.normpath(save_folder))
        subprocess.check_output(
            ['tar', '-czf', save_folder + '.tar.gz',
             '-C', up_to_sf, sf])
