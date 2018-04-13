from shapely.geometry import Point, box
from shapely import affinity
import numpy as np
import random
from PIL import Image
import aggdraw
from enum import Enum
from tqdm import trange


DIM = 64
X_MIN, X_MAX = (8, 48)
SIZE_MIN, SIZE_MAX = (3, 9)


TWOFIVEFIVE = np.float32(255)


SHAPES = ['circle', 'square', 'rectangle', 'ellipse']
COLORS = ['red', 'blue', 'green', 'yellow', 'white', 'gray']
BRUSHES = {c: aggdraw.Brush(c) for c in COLORS}
PENS = {c: aggdraw.Pen(c) for c in COLORS}


class ShapeSpec(Enum):
    SHAPE = 0
    COLOR = 1
    BOTH = 2


class ConfigProps(Enum):
    SHAPE_1_COLOR = 0
    SHAPE_1_SHAPE = 1
    SHAPE_2_COLOR = 2
    SHAPE_2_SHAPE = 3
    RELATION_DIR = 4


SHAPE_SPECS = list(ShapeSpec)


def rand_size():
    return random.randrange(SIZE_MIN, SIZE_MAX)


def rand_pos():
    return random.randrange(X_MIN, X_MAX)


class Shape:
    def __init__(self, color=None):
        if color is None:
            self.color = random.choice(COLORS)
        else:
            self.color = color
        self.x = rand_pos()
        self.y = rand_pos()
        self.init_shape()

    def draw(self, image):
        image.draw.polygon(self.coords, PENS[self.color])

    def intersects(self, oth):
        return self.shape.intersects(oth.shape)


class Ellipse(Shape):
    def init_shape(self, min_skew=1.5):
        self.dx = rand_size()
        # Dy must be at least 1.6x dx, to remove ambiguity with circle
        bigger = int(self.dx * min_skew)
        if bigger >= SIZE_MAX:
            smaller = int(self.dx / min_skew)
            assert smaller > SIZE_MIN, ("{} {}".format(smaller, self.dx))
            self.dy = random.randrange(SIZE_MIN, smaller)
        else:
            self.dy = random.randrange(bigger, SIZE_MAX)
        if random.random() < 0.5:
            # Switch dx, dy
            self.dx, self.dy = self.dy, self.dx

        shape = Point(self.x, self.y).buffer(1)
        shape = affinity.scale(shape, self.dx, self.dy)
        shape = affinity.rotate(shape, random.randrange(360))
        self.shape = shape

        #  self.coords = [int(x) for x in self.shape.bounds]
        self.coords = np.round(np.array(self.shape.boundary).astype(np.int))
        #  print(len(np.array(self.shape.convex_hull)))
        #  print(len(np.array(self.shape.convex_hull.boundary)))
        #  print(len(np.array(self.shape.exterior)))
        self.coords = np.unique(self.coords, axis=0).flatten()


class Circle(Ellipse):
    def init_shape(self):
        self.r = rand_size()
        self.shape = Point(self.x, self.y).buffer(self.r)
        self.coords = [int(x) for x in self.shape.bounds]

    def draw(self, image):
        image.draw.ellipse(self.coords, BRUSHES[self.color])


class Rectangle(Shape):
    def init_shape(self):
        self.dx = rand_size()
        self.dy = rand_size()
        shape = box(self.x, self.y, self.x + self.dx, self.y + self.dy)
        # Rotation
        shape = affinity.rotate(shape, random.randrange(90))
        self.shape = shape

        # Get coords
        self.coords = np.round(np.array(self.shape.exterior.coords)[:-1].flatten()).astype(np.int).tolist()

    def draw(self, image):
        image.draw.polygon(self.coords, BRUSHES[self.color], PENS[self.color])


class Square(Rectangle):
    def init_shape(self):
        self.size = rand_size()
        shape = box(self.x, self.y, self.x + self.size, self.y + self.size)
        # Rotation
        shape = affinity.rotate(shape, random.randrange(90))
        self.shape = shape

        # Get coords
        self.coords = np.round(np.array(self.shape.exterior.coords)[:-1].flatten()).astype(np.int).tolist()


SHAPE_IMPLS = {
    'circle': Circle,
    'ellipse': Ellipse,
    'square': Square,
    'rectangle': Rectangle,
    # TODO: Triangle, semicircle
}


class I:
    def __init__(self):
        self.image = Image.new('RGB', (DIM, DIM))
        #  self.draw = ImageDraw.Draw(self.image)
        self.draw = aggdraw.Draw(self.image)

    def draw_shapes(self, shapes, flush=True):
        for shape in shapes:
            shape.draw(self)
        if flush:
            self.draw.flush()

    def show(self):
        self.image.show()
        #  self.image.resize((64, 64), Image.ANTIALIAS).show()

    def array(self):
        return np.array(self.image, dtype=np.uint8)

    def float_array(self):
        return np.divide(np.array(self.image), TWOFIVEFIVE)

    def save(self, path, filetype='PNG'):
        self.image.save(path, filetype)


def random_shape():
    return random.choice(SHAPES)


def random_color():
    return random.choice(COLORS)


def random_shape_from_spec(spec):
    color = None
    shape = None
    if spec == ShapeSpec.SHAPE:
        shape = random_shape()
    elif spec == ShapeSpec.COLOR:
        color = random_color()
    elif spec == ShapeSpec.BOTH:
        shape = random_shape()
        color = random_color()
    else:
        raise ValueError("Unknown spec {}".format(spec))
    return (color, shape)


def random_config():
    # 0 -> only shape specified
    # 1 -> only color specified
    # 2 -> only both specified
    shape_1_spec = ShapeSpec(random.randrange(3))
    shape_2_spec = ShapeSpec(random.randrange(3))
    shape_1 = random_shape_from_spec(shape_1_spec)
    shape_2 = random_shape_from_spec(shape_2_spec)
    if shape_1 == shape_2:
        return random_config()
    relation = random.randrange(2)
    relation_dir = random.randrange(2)
    return [shape_1, shape_2], None, relation, relation_dir


def add_shape_from_spec(spec, shapes=None, attempt=1, max_attempts=3):
    if attempt > max_attempts:
        return None
    color, shape_ = spec
    if shape_ is None:
        shape_ = random_shape()
    shape = SHAPE_IMPLS[shape_](color=color)
    if shapes is not None:
        for oth in shapes:
            if shape.intersects(oth):
                return add_shape_from_spec(spec, shapes, attempt=attempt + 1,
                                           max_attempts=max_attempts)
        shapes.append(shape)
        return shape
    return shape


def new_color(existing_color):
    new_c = existing_color
    while new_c == existing_color:
        new_c = random.choice(COLORS)
    return new_c


def new_shape(existing_shape):
    new_s = existing_shape
    while new_s == existing_shape:
        new_s = random.choice(SHAPES)
    return new_s


def invalidate(config):
    # Invalidate by randomly choosing one property to change:
    ((shape_1_color, shape_1_shape),
     (shape_2_color, shape_2_shape)), extra_shape_specs, relation, relation_dir = config
    properties = []
    if shape_1_color is not None:
        properties.append(ConfigProps.SHAPE_1_COLOR)
    if shape_1_shape is not None:
        properties.append(ConfigProps.SHAPE_1_SHAPE)
    if shape_2_color is not None:
        properties.append(ConfigProps.SHAPE_2_COLOR)
    if shape_2_shape is not None:
        properties.append(ConfigProps.SHAPE_2_SHAPE)
    # Don't invalidate relation dir, since it can be invalidated separately
    #  properties.append(ConfigProps.RELATION_DIR)
    # Randomly select property to invalidate
    # TODO: Support for invalidating multiple properties
    invalid_prop = random.choice(properties)

    if invalid_prop == ConfigProps.SHAPE_1_COLOR:
        return ((new_color(shape_1_color), shape_1_shape),
                (shape_2_color, shape_2_shape)), extra_shape_specs, relation, relation_dir
    elif invalid_prop == ConfigProps.SHAPE_1_SHAPE:
        return ((shape_1_color, new_shape(shape_1_shape)),
                (shape_2_color, shape_2_shape)), extra_shape_specs, relation, relation_dir
    elif invalid_prop == ConfigProps.SHAPE_2_COLOR:
        return ((shape_1_color, shape_1_shape),
                (new_color(shape_2_color), shape_2_shape)), extra_shape_specs, relation, relation_dir
    elif invalid_prop == ConfigProps.SHAPE_2_SHAPE:
        return ((shape_1_color, shape_1_shape),
                (shape_2_color, new_shape(shape_2_shape))), extra_shape_specs, relation, relation_dir
    elif invalid_prop == ConfigProps.RELATION_DIR:
        return ((shape_1_color, shape_1_shape),
                (shape_2_color, shape_2_shape)), extra_shape_specs, relation, 1 - relation_dir
    else:
        raise RuntimeError


def fmt_config(config):
    (s1, s2), extra, relation, relation_dir = config
    if relation == 0:
        if relation_dir == 0:
            rel_txt = 'left'
        else:
            rel_txt = 'right'
    else:
        if relation_dir == 0:
            rel_txt = 'above'
        else:
            rel_txt = 'below'
    if s1[0] is None:
        s1_0_txt = ''
    else:
        s1_0_txt = s1[0]
    if s1[1] is None:
        s1_1_txt = 'shape'
    else:
        s1_1_txt = s1[1]
    if s2[0] is None:
        s2_0_txt = ''
    else:
        s2_0_txt = s2[0]
    if s2[1] is None:
        s2_1_txt = 'shape'
    else:
        s2_1_txt = s2[1]
    return '{} {} {} {} {}'.format(s1_0_txt, s1_1_txt, rel_txt.upper(), s2_0_txt, s2_1_txt)



if __name__ == '__main__':
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

    parser = ArgumentParser(
        description='Fast ShapeWorld',
        formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument('--n', type=int, default=100, help='Number of instances')
    parser.add_argument('--wpi', type=int, default=10, help='Worlds per instance')

    args = parser.parse_args()

    imgs = np.zeros((args.n, args.wpi, 64, 64, 3), dtype=np.uint8)
    labels = np.zeros((args.n, args.wpi), dtype=np.uint8)
    configs = []

    for n in trange(args.n):
        # Get shapes and relations
        config = random_config()
        configs.append(config)
        for wpi in range(args.wpi):
            label = int(random.random() < 0.75)
            new_config = config if label else invalidate(config)
            print("\t", new_config)
            (ss1, ss2), extra_shape_specs, relation, relation_dir = new_config
            s1 = add_shape_from_spec(ss1)
            s2 = add_shape_from_spec(ss2)

            attempts = 0
            while attempts < 5:
                s2 = add_shape_from_spec(ss2)
                if not s2.intersects(s1):
                    break
            else:
                # Failed
                raise RuntimeError

            # Figure out if relation dir is correct.
            s1_centroid = s1.shape.centroid
            s2_centroid = s2.shape.centroid
            if relation == 0:
                if s1_centroid.x > s2_centroid.x:
                    this_rel_dir = 1
                else:
                    this_rel_dir = 0
            elif relation == 1:
                if s1_centroid.y > s2_centroid.y:
                    this_rel_dir = 1
                else:
                    this_rel_dir = 0

            if this_rel_dir != relation_dir:
                # Make sure label is false, even if the config was invalidated
                # in the first place.
                label = 0

            # Sample world type:
            # Both well specified - one well specified - neither well specified
            # Based on that, generate random shape reqs
            img = I()
            img.draw_shapes([s1, s2])
            imgs[n, wpi] = img.array()
            labels[n, wpi] = label
            #  img.show()

    for instance_idx, (instance, instance_labels) in enumerate(zip(imgs, labels)):
        for world_idx, (world, label) in enumerate(zip(instance, instance_labels)):
            Image.fromarray(world).save('test/{}_{}.png'.format(instance_idx, world_idx))

    with open('test/index.html', 'w') as f:
        f.write(
            '''
            <!DOCTYPE html>
            <html>
            <head>
            <title>Shapeworld Fast</title>
            <style>
            img {{
                padding: 10px;
            }}
            img.yes {{
                background-color: green;
            }}
            img.no {{
                background-color: red;
            }}
            </style>
            </head>
            <body>
            {}
            </body>
            </html>
            '''.format(
                ''.join(
                    '<h1>{}</h1><p>{}</p>'.format(fmt_config(config), ''.join(
                        '<img src="{}_{}.png" class="{}">'.format(instance_idx, world_idx, 'yes' if label else 'no') for world_idx, (world, label) in enumerate(zip(instance, instance_labels))
                    )) for instance_idx, (instance, instance_labels, config) in enumerate(zip(imgs, labels, configs))
                )
            )
        )
    np.savez_compressed('test.npz', imgs=imgs, labels=labels)
