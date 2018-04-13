from shapely.geometry import Point, box
from shapely import affinity
import numpy as np
import random
from PIL import Image
import aggdraw
import tensorflow as tf


DIM = 64
X_MIN, X_MAX = (8, 56)
SIZE_MIN, SIZE_MAX = (4, 10)


TWOFIVEFIVE = np.float32(255)


COLORS = ['red', 'blue', 'green', 'yellow', 'white']
#  COLORS = ['red', 'blue', 'green', 'yellow', 'white']
#  COLORS = ['magenta']
BRUSHES = {c: aggdraw.Brush(c) for c in COLORS}
PENS = {c: aggdraw.Pen(c) for c in COLORS}


def rand_size():
    return random.randrange(SIZE_MIN, SIZE_MAX)

def rand_pos():
    return random.randrange(X_MIN, X_MAX)


class Shape:
    def __init__(self):
        self.color = random.choice(COLORS)
        self.x = rand_pos()
        self.y = rand_pos()
        self.init_shape()

    def draw(self, image):
        image.draw.polygon(self.coords, PENS[self.color])


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


N = 100
WORLDS_PER_INSTANCE = 10


imgs = np.zeros((N, WORLDS_PER_INSTANCE, 64, 64, 3), dtype=np.uint8)
print(imgs.nbytes)
import sys
print(sys.getsizeof(imgs))
labels = np.zeros((N, WORLDS_PER_INSTANCE), dtype=np.uint8)

from tqdm import trange
for n in trange(N):
    for wpi in range(WORLDS_PER_INSTANCE):
        img = I()
        shapes = [Square() for _ in range(2)] + [Ellipse() for _ in range(5)]
        img.draw_shapes(shapes)
        imgs[n, wpi] = img.array()
        labels[n, wpi] = random.randrange(2)

print(imgs.nbytes)
print(labels.nbytes)
np.savez_compressed('hello.npz', imgs=imgs, labels=labels)
#  shapes = [Ellipse()]
#  print(shapes[0].coords)

#  img.draw_shapes(shapes)
#  img.show()
