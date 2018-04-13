from shapely.geometry import Point, box
from shapely import affinity
import numpy as np
import random
from PIL import Image
import aggdraw
from enum import Enum
from tqdm import trange


DIM = 64
X_MIN, X_MAX = (8, 56)
SIZE_MIN, SIZE_MAX = (4, 10)


TWOFIVEFIVE = np.float32(255)


SHAPES = ['circle', 'square', 'ellipse']
COLORS = ['red', 'blue', 'green', 'yellow', 'white']
#  COLORS = ['red', 'blue', 'green', 'yellow', 'white']
#  COLORS = ['magenta']
BRUSHES = {c: aggdraw.Brush(c) for c in COLORS}
PENS = {c: aggdraw.Pen(c) for c in COLORS}


class ShapeSpec(Enum):
    SHAPE = 0
    COLOR = 1
    BOTH = 2


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
    def init_shape(self):
        self.dx = rand_size()
        self.dy = rand_size()

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


class Square(Shape):
    def init_shape(self):
        self.size = rand_size()
        shape = box(self.x, self.y, self.x + self.size, self.y + self.size)
        # Rotation
        shape = affinity.rotate(shape, random.randrange(90))
        self.shape = shape

        # Get coords
        self.coords = np.round(np.array(self.shape.exterior.coords)[:-1].flatten()).astype(np.int).tolist()

    def draw(self, image):
        image.draw.polygon(self.coords, BRUSHES[self.color])


SHAPE_IMPLS = {
    'circle': Circle,
    'square': Square,
    'ellipse': Ellipse,
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
    shape = None
    color = None
    if spec == ShapeSpec.SHAPE:
        shape = random_shape()
    elif spec == ShapeSpec.COLOR:
        color = random_color()
    elif spec == ShapeSpec.BOTH:
        shape = random_shape()
        color = random_color()
    else:
        raise ValueError("Unknown spec {}".format(spec))
    return (shape, color)


def random_config():
    # 0 -> only shape specified
    # 1 -> only color specified
    # 2 -> only both specified
    shape_1_spec = random.choice(SHAPE_SPECS)
    shape_2_spec = random.choice(SHAPE_SPECS)
    shape_1 = random_shape_from_spec(shape_1_spec)
    shape_2 = random_shape_from_spec(shape_2_spec)
    if shape_1 == shape_2:
        return random_config()
    relation = random.randrange(2)
    relation_dir = random.randrange(2)
    return ([shape_1, shape_2], relation, relation_dir)


def add_shape_from_spec(spec, shapes, attempt=1, max_attempts=3):
    if attempt > max_attempts:
        return False
    shape_, color = spec
    if shape_ is None:
        shape_ = random_shape()
    shape = SHAPE_IMPLS[shape_](color=color)
    for oth in shapes:
        if shape.intersects(oth):
            print("Collision, retrying (attempt {})".format(attempt))
            return add_shape_from_spec(spec, shapes, attempt=attempt + 1,
                                       max_attempts=max_attempts)
    shapes.append(shape)
    return True


if __name__ == '__main__':
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

    parser = ArgumentParser(
        description='Fast ShapeWorld',
        formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument('--n', type=int, default=100, help='Number of instances')
    parser.add_argument('--wpi', type=int, default=10, help='Worlds per instance')

    args = parser.parse_args()

    N = 100
    WORLDS_PER_INSTANCE = 10

    imgs = np.zeros((args.n, args.wpi, 64, 64, 3), dtype=np.uint8)
    labels = np.zeros((args.n, args.wpi), dtype=np.uint8)

    for n in trange(args.n):
        for wpi in range(args.wpi):
            shape_specs, relation, relation_dir = random_config()
            shapes = []
            for shape_spec in shape_specs:
                add_shape_from_spec(shape_spec, shapes)
            continue
            # Sample world type:
            # Both well specified - one well specified - neither well specified
            # Based on that, generate random shape reqs
            img = I()
            img.draw_shapes(shapes)
            imgs[n, wpi] = img.array()
            labels[n, wpi] = random.randrange(2)

    np.savez_compressed('test.npz', imgs=imgs, labels=labels)
