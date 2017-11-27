"""
Generate random shapeworld data
"""

from collections import namedtuple
import numpy as np
from shapeworld.dataset import CaptionAgreementDataset
from shapeworld.generators import RandomAttributesGenerator
from shapeworld.captioners import ExistentialCaptioner, RegularTypeCaptioner, RelationCaptioner
from shapeworld.world import World
import os
from tqdm import trange
from PIL import Image

random = np.random.RandomState()


def invert(d):
    return {v: k for k, v in d.items()}


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

Scene = namedtuple('Scene', ['images', 'target'])
Shape = namedtuple('Shape', ['name', 'color', 'size', 'center', 'rotation'])
Point = namedtuple('Point', ['x', 'y'])

TEXTURES = ['solid']
TEXTURES_MAP = invert(dict(enumerate(TEXTURES)))

RELATIONS = ['x-rel', 'y-rel']


def comp_obj(entity_model, obj):
    # e.g. (square, red, solid)
    for i, prop in enumerate(['shape', 'color', 'texture']):
        if prop in entity_model and entity_model[prop]['value'] != obj[i]:
            return False
    return True


def gen_example_data(n, n_images=2, max_shapes=5, target='simple'):
    """
    Generate a list of Scenes
    where `target` is either 0 or 1 and image{1,2} is a list of Shapes.
    """
    return [gen_scene(n_images, max_shapes, target=target) for _ in range(n)]


def gen_scene(n_images, max_shapes, target='simple'):
    """Generate a single target/distractor scene."""
    images = [gen_random_shapes(max_shapes) for _ in range(n_images)]
    if target == 'simple':
        # Super simple target: whichever image has the most shapes
        target = max(enumerate(images), key=lambda t: len(t[1]))[0]
    elif target == 'white':
        n_white_0 = sum(i.color == 'white' for i in images[0])
        n_white_1 = sum(i.color == 'white' for i in images[1])
        if n_white_0 > n_white_1:
            target = 1
        else:
            target = 0
    else:
        raise NotImplementedError("Unknown target {}".format(target))
    return Scene(images=images, target=target)


def gen_random_shapes(max_shapes):
    """Generate a list of random shapes."""
    n_items = random.randint(max_shapes) + 1
    return [gen_random_shape() for _ in range(n_items)]


def gen_random_shape():
    """Generate a random shape."""
    return Shape(
        name=random.choice(NAMES),
        color=random.choice(COLORS),
        size=random_point(),
        center=random_point(),
        rotation=random.rand())


def flatten_scenes(scenes):
    """Flatten shapes in scenes"""
    return list(map(flatten_scene, scenes))


def flatten_scene(scene):
    """Flatten a to make it tf-compatible"""
    return Scene(
        images=list(map(flatten_shapes, scene.images)), target=scene.target)


def flatten_shapes(shapes):
    return list(map(flatten_shape, shapes))


def flatten_shape(shape):
    """Return structured array representation of shape"""
    names_onehot = [0 for _ in NAMES]
    names_onehot[NAMES_MAP[shape.name]] = 1
    colors_onehot = [0 for _ in COLORS]
    colors_onehot[COLORS_MAP[shape.color]] = 1
    reals = [
        shape.size.x, shape.size.y, shape.center.x, shape.center.y,
        shape.rotation
    ]
    return names_onehot + colors_onehot + reals


def random_point():
    return Point(x=random.rand(), y=random.rand())


def most_items(scene):
    """Given a Scene, return the max number of items in an image"""
    return max(map(len, scene.images))


def extract_envs_and_labels(scenes, n_images, max_shapes, n_attrs):
    """
    Given a list of scenes, return a list of tf-compatible feature reps and
    labels
    """
    n_scenes = len(scenes)

    envs = np.zeros((n_scenes, n_images, max_shapes * n_attrs))
    for scene_i, scene in enumerate(scenes):
        for image_i, image in enumerate(scene.images):
            global_shape_i = 0
            for shape in image:
                for shape_prop in shape:
                    envs[scene_i, image_i, global_shape_i] = shape_prop
                    global_shape_i += 1

    labels = np.zeros((n_scenes, n_images))
    for scene_i, scene in enumerate(scenes):
        labels[scene_i, scene.target] = 1

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


class SpatialExtraSimple(CaptionAgreementDataset):
    """
    A dataset that generates target/distractor pairs with common objects and
    simple relations.
    """

    def __init__(self,
                 combinations=None,
                 relation=None,
                 relation_dir=None,
                 target_i=None):
        # Randomly sample combinations and relations if not provided
        if combinations is None:
            combinations = (None, None)
            while combinations[0] == combinations[1]:
                combinations = tuple(random_objects(2))
        if relation is None:
            relation = random.choice(RELATIONS)

        self.shapes = combinations
        self.relation = relation

        if relation_dir is None:
            self.relation_dir = random.choice([1, -1])
        if target_i is None:
            self.target_i = random.randint(len(self.shapes))
            self.target_obj = self.shapes[self.target_i]

        assert relation in ('x-rel', 'y-rel'), "Invalid relation"

        vocabulary = sorted([
            '.', 'a', 'above', 'an', 'below', 'blue', 'circle', 'cross',
            'cyan', 'ellipse', 'gray', 'green', 'is', 'left', 'magenta', 'of',
            'pentagon', 'rectangle', 'red', 'right', 'semicircle', 'shape',
            'square', 'the', 'to', 'triangle', 'yellow'
        ])

        world_generator = FixedWorldGenerator(
            entity_counts=[2],
            validation_combinations=combinations,
            test_combinations=combinations,
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

        for imgcm in zip(sw_data['world'], sw_data['caption_model']):
            img, cm = imgcm
            img = sw_arr_to_img(img)

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
                    print("Warning: both targets ambiguous:", cm)
                    continue
                cap_target_i = opps_is_target.index(False)
            else:
                cap_target_i = combs_is_target.index(True)

            if self.relation_dir == cap_relation_dir:
                if self.target_i == cap_target_i:
                    targets.append((img, cm))
                else:
                    distractors.append((img, cm))
            else:
                # Since the relation direction is flipped,
                # the targets are flipped
                if self.target_i == cap_target_i:
                    distractors.append((img, cm))
                else:
                    targets.append((img, cm))

        return targets, distractors

    def generate(self, max, noise_range=0.0, n_distractors=1):
        targets, distractors = self.generate_targets_distractors(
            max, noise_range=noise_range)
        return self.combine_targets_distractors(
            targets, distractors, n_distractors=n_distractors)

    def combine_targets_distractors(self,
                                    targets,
                                    distractors,
                                    n_distractors=2):
        return targets, distractors

    def to_html(self, targets, distractors, save_dir='test'):
        """
        TODO: change this so that it accepts the combined 3way
        images
        """
        html_str = """
            <!DOCTYPE html>
            <html>
                <head>
                    <title>temp</title>
                    <style>
                    img {
                        margin: 5px;
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
        html_str += "<h1>Targets</h1>"

        for i, world in enumerate(targets):
            img, cm = world
            world_name = 'world-{}-target.bmp'.format(i)
            html_str += "<img src={}></img>".format(world_name)
            img.save(os.path.join(save_dir, world_name))

        html_str += "<h1>Distractors</h1>"
        for i, world in enumerate(distractors):
            img, cm = world
            world_name = 'world-{}-distractor.bmp'.format(i)
            html_str += "<img src={}></img>".format(world_name)
            img.save(os.path.join(save_dir, world_name))

        html_str += "</body></html>"

        with open(os.path.join(save_dir, 'index.html'), 'w') as fout:
            fout.write(html_str)
            fout.write('\n')

    @classmethod
    def to_html_many(cls, datas, save_dir='test'):
        raise NotImplementedError


if __name__ == "__main__":
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter,
        description='Generate spatial datasets')
    parser.add_argument(
        '-n', default=1, type=int, help='# dataset configs to produce')
    parser.add_argument(
        '--each', default=100, type=int, help='# (max) images per config')

    args = parser.parse_args()

    for _ in trange(args.n):
        dataset = SpatialExtraSimple()
        targets, distractors = dataset.generate(args.each)
        dataset.to_html(targets, distractors, save_dir='test')
