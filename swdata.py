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
from itertools import cycle

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

TEXTURES = ['solid']
TEXTURES_MAP = invert(dict(enumerate(TEXTURES)))

Scene = namedtuple('Scene', ['worlds', 'labels', 'relation', 'relation_dir'])
SWorld = namedtuple('SWorld', ['image', 'shapes'])
Shape = namedtuple('Shape',
                   ['name', 'color', 'size', 'center', 'rotation', 'texture'])
Point = namedtuple('Point', ['x', 'y'])
Color = namedtuple('Color', ['color', 'shade'])

RELATIONS = ['x-rel', 'y-rel']


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


def flatten_shapes(shapes):
    return list(map(flatten_shape, shapes))


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
    return names_onehot + colors_onehot + textures_onehot + reals


def extract_envs_and_labels(scenes, n_images, max_shapes, n_attrs):
    """
    Given a list of scenes, return a list of tf-compatible feature reps and
    labels
    """
    n_scenes = len(scenes)

    envs = np.zeros((n_scenes, n_images, max_shapes * n_attrs))
    labels = np.zeros((n_scenes, n_images))

    for scene_i, scene in enumerate(scenes):
        for world_i, wl in enumerate(zip(scene.worlds, scene.labels)):
            world, label = wl
            global_shape_i = 0
            for shape in world.shapes:
                for shape_prop in shape:
                    envs[scene_i, world_i, global_shape_i] = shape_prop
                    global_shape_i += 1
            labels[scene_i, world_i] = label

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
                 n_distractors=1):
        targets, distractors = self.generate_targets_distractors(
            max, noise_range=noise_range)
        return self.combine_targets_distractors(
            targets, distractors,
            n_targets=n_targets, n_distractors=n_distractors)

    def combine_targets_distractors(self,
                                    targets,
                                    distractors,
                                    n_targets=2,
                                    n_distractors=1):
        n_scenes = max_scenes(len(targets), len(distractors),
                              n_targets, n_distractors)

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
        html_str = """
            <!DOCTYPE html>
            <html>
                <head>
                    <title>temp</title>
                    <style>
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
        train = dataset.generate(args.each)
        dataset.to_html(train, save_dir='test')
