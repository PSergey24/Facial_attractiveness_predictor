import dlib
import numpy as np
from math import sqrt, acos, degrees


class VectorTools:

    @staticmethod
    def angle_between(v1, v2, cosinus=False):
        x1, y1 = v1[0], v1[1]
        x2, y2 = v2[0], v2[1]

        def scalar(x1, y1, x2, y2):
            return x1 * x2 + y1 * y2

        def module(x, y):
            return sqrt(x ** 2 + y ** 2)

        cos = scalar(x1, y1, x2, y2) / (module(x1, y1) * module(x2, y2))
        return degrees(acos(cos)) if cosinus is False else cos

    @staticmethod
    def get_vector(x1, y1, x2, y2):
        return x1 - x2, y1 - y2

    @staticmethod
    def euclid_distance(a, b):
        return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


# convert data from dlib to OpenCV and back
class CvTools:
    @staticmethod
    def shape_to_np(shape):
        return [(shape.part(i).x, shape.part(i).y) for i in range(0, shape.num_parts)]

    @staticmethod
    def rect_to_bb(rect):
        x = rect.left()
        y = rect.top()
        w = rect.right() - x
        h = rect.bottom() - y
        return x, y, w, h

    @staticmethod
    def bb_to_rect(x, y, w, h):
        return dlib.rectangle(x, y, w, h)
