import cv2
import numpy as np
import math
from utils import utils
from utils import Point

class Shape:
    def __init__(self, points=[]):
        self.pts = points

        self.Ypoints = []

    def add_point(self, point):

        self.pts.append(point)

    def placeModel(self):
        points = []
        x_offset = 1360
        y_offset = 740
        for point in self.pts:
            point.x = point.x + x_offset
            point.y = point.y + y_offset

            points.append(point)
        self.pts = points



    @staticmethod
    def from_vector(vec):
        s = Shape([])
        for i,j in np.reshape(vec, (-1,2)):
            s.add_point(Point(i, j))
        return s

    def align_to_shape(self, s, w):
        # s = first shape
        # w = weight matrix

        p = self.get_alignment_params(s, w)
        return self.apply_params_to_shape(p)

    # TODO
    def get_normal_to_point(self, p_num):
        # Normal to first point
        x = 0; y = 0; mag = 0
        if p_num == 0:
          x = self.pts[1].x - self.pts[0].x
          y = self.pts[1].y - self.pts[0].y
        # Normal to last point
        elif p_num == len(self.pts)-1:
          x = self.pts[-1].x - self.pts[-2].x
          y = self.pts[-1].y - self.pts[-2].y
        # Must have two adjacent points, so...
        else:
          x = self.pts[p_num+1].x - self.pts[p_num-1].x
          y = self.pts[p_num+1].y - self.pts[p_num-1].y
        mag = math.sqrt(x**2 + y**2)
        return (-y/mag, x/mag)

    def get_alignment_params(self, s, w):
        """ Gets the parameters required to align the shape to the given shape
        using the weight matrix w.  This applies a scaling, transformation and
        rotation to each point in the shape to align it as closely as possible
        to the shape.
        This relies on some linear algebra which we use numpy to solve.
        [ X2 -Y2   W   0][ax]   [X1]
        [ Y2  X2   0   W][ay] = [Y1]
        [ Z    0  X2  Y2][tx]   [C1]
        [ 0    Z -Y2  X2][ty]   [C2]
        We want to solve this to find ax, ay, tx, and ty
        :param shape: The shape to align to
        :param w: The weight matrix
        :return x: [ax, ay, tx, ty]
        """


        X1 = s.__get_X(w)
        X2 = self.__get_X(w)
        Y1 = s.__get_Y(w)
        Y2 = self.__get_Y(w)
        Z = self.__get_Z(w)
        W = sum(w)
        C1 = self.__get_C1(w, s)
        C2 = self.__get_C2(w, s)

        a = np.array([[ X2, -Y2,   W,  0],
                      [ Y2,  X2,   0,  W],
                      [  Z,   0,  X2, Y2],
                      [  0,   Z, -Y2, X2]])

        b = np.array([X1, Y1, C1, C2])
        # Solve equations
        # result is [ax, ay, tx, ty]
        return np.linalg.solve(a, b)

    def apply_params_to_shape(self, p):
        new = Shape([]) # TODO!!!!
        # For each point in current shape


        for pt in self.pts:
          new_x = (p[0]*pt.x - p[1]*pt.y) + p[2]
          new_y = (p[1]*pt.x + p[0]*pt.y) + p[3]
          new.add_point(Point(new_x, new_y)) # TODO!!!!
        return new

    def __get_X(self, w):



        return sum([w[i]*self.pts[i].x for i in range(len(self.pts))])
    def __get_Y(self, w):
        return sum([w[i]*self.pts[i].y for i in range(len(self.pts))])
    def __get_Z(self, w):
        return sum([w[i]*(self.pts[i].x**2+self.pts[i].y**2) for i in range(len(self.pts))])
    def __get_C1(self, w, s):
        return sum([w[i]*(s.pts[i].x*self.pts[i].x + s.pts[i].y*self.pts[i].y) \
            for i in range(len(self.pts))])
    def __get_C2(self, w, s):
        return sum([w[i]*(s.pts[i].y*self.pts[i].x - s.pts[i].x*self.pts[i].y) \
            for i in range(len(self.pts))])

    def get_vector(self):
        vec = np.zeros((320, 2))
        for i in range(len(self.pts)):
          vec[i,:] = [self.pts[i].x, self.pts[i].y]
        return vec.flatten()


    def vecToPoints(self, vec):
        points = []

        counter = 0
        for coord in vec:

            if(counter % 2 == 0):
                x = coord
            else:
                y = coord
                point = Point(x,y)
                points.append(point)
            counter = counter +1
        return points

    def drawShape(self):

        test_images = ['_Data/Radiographs/01.tif']

        for test_image in test_images:
            matrix = cv2.imread(test_image)

            utils.drawShape(matrix,self.pts)

    def pointsToVec(self):
        #vec = np.empty((640,))

        vec = np.array([])

        for point in self.pts:
            point_arr = [point.x,point.y]
            vec = np.append(vec,point_arr)

        return vec


