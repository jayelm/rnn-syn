"""
Generate random shapeworld data
"""

from collections import namedtuple
import numpy as np

random = np.random.RandomState(1)

COLORS = [
    'black',
    'red',
    'green',
    'blue',
    'yellow',
    'magenta',
    'cyan',
    'white',
]
COLORS_MAP = {v: k for k, v in dict(enumerate(COLORS)).items()}

NAMES = ['semicircle', 'circle', 'cross', 'pentagon', 'square', 'oval']
NAMES_MAP = {v: k for k, v in dict(enumerate(NAMES)).items()}

Scene = namedtuple('Scene', ['images', 'target'])
Shape = namedtuple('Shape', ['name', 'color', 'size', 'center', 'rotation'])
Point = namedtuple('Point', ['x', 'y'])


def gen_example_data(n, n_images=2, max_shapes=5, target='simple'):
    """
    Generate a list of Scenes
    where `target` is either 0 or 1 and image{1,2} is a list of Shapes.
    """
    return [gen_scene(n_images, max_shapes, target=target)
            for _ in range(n)]


def gen_scene(n_images, max_shapes, target='simple'):
    """Generate a single target/distractor scene."""
    images = [gen_random_shapes(max_shapes) for _ in range(n_images)]
    if target == 'simple':
        # Super simple target: whichever one has more shapes
        if len(images[0]) > len(images[1]):
            target = 0
        else:
            target = 1
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
    return Scene(images=list(map(flatten_shapes, scene.images)),
                 target=scene.target)


def flatten_shapes(shapes):
    return list(map(flatten_shape, shapes))


def flatten_shape(shape):
    """Return structured array representation of shape"""
    names_onehot = [0 for _ in NAMES]
    names_onehot[NAMES_MAP[shape.name]] = 1
    colors_onehot = [0 for _ in COLORS]
    colors_onehot[COLORS_MAP[shape.color]] = 1
    reals = [shape.size.x, shape.size.y, shape.center.x,
             shape.center.y, shape.rotation]
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
