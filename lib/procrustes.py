from scipy.spatial import procrustes
import numpy as np
import sys
from Shape import Shape

class ProcusterAnalysis:
    # returns new images array with landmarks scaled, aligned and transformed
    def __init__(self,images):
        self.images = images

        self.shapes = []


        self.getShapes()

        print('original point',self.shapes[1].pts[0].y)

        self.w = self.__create_weight_matrix(self.shapes)

        self.__procrustes(self.shapes)


    def getShapes(self):
        for image in self.images:
            shape = Shape(image.points)
            self.shapes.append(shape)


    def getTeethMatrix(self,teeth):
        teethMatrix = np.empty((0,80),np.uint8)

        for tooth in teeth:
            landmarksVector = []
            for landmark in tooth['landmarks']:

                    landmarksVector.append(landmark['x'])
                    landmarksVector.append(landmark['y'])
            teethMatrix = np.vstack([teethMatrix,landmarksVector])
        return teethMatrix

    def __create_weight_matrix(self, shapes):
        """ Private method to produce the weight matrix which corresponds
        to the training shapes
        :param shapes: A list of Shape objects
        :return w: The matrix of weights produced from the shapes
        """
        # Return empty matrix if no shapes
        if not shapes:
          return np.array()
        # First get number of points of each shape
        num_pts = 320

        # We need to find the distance of each point to each
        # other point in each shape.
        distances = np.zeros((len(shapes), num_pts, num_pts))
        for s, shape in enumerate(shapes):

            for k in range(num_pts):
                for l in range(num_pts):
                    distances[s, k, l] = shape.pts[k].dist(shape.pts[l])

        # Create empty weight matrix
        w = np.zeros(num_pts)
        # calculate range for each point
        for k in range(num_pts):
          for l in range(num_pts):
            # Get the variance in distance of that point to other points
            # for all shapes
            w[k] += np.var(distances[:, k, l])
        # Invert weights
        self.w = 1/w
        return self.w

    def __get_mean_shape(self, shapes):
        s = shapes[0].pointsToVec()


        for shape in shapes[1:]:
            s = s + shape.pointsToVec()
        vec = s / len(shapes)

        shape = Shape([])
        shape.points = shape.vecToPoints(vec)
        shape.pts = shape.vecToPoints(vec)

        return shape

    def __procrustes(self, shapes):
        """ This function aligns all shapes passed as a parameter by using
        Procrustes analysis
        :param shapes: A list of Shape objects
        """
        # First rotate/scale/translate each shape to match first in set
        shapes[1:] = [s.align_to_shape(shapes[0], self.w) for s in shapes[1:]]

        # Keep hold of a shape to align to each iteration to allow convergence
        a = shapes[0]
        trans = np.zeros((4, len(shapes)))
        converged = False
        current_accuracy = sys.maxint
        while not converged:
            # Now get mean shape
            mean = self.__get_mean_shape(shapes)
            # Align to shape to stop it diverging
            mean = mean.align_to_shape(a, self.w)
            # Now align all shapes to the mean
            for i in range(len(shapes)):
                # Get transformation required for each shape
                trans[:, i] = shapes[i].get_alignment_params(mean, self.w)
                # Apply the transformation
                shapes[i] = shapes[i].apply_params_to_shape(trans[:,i])

            # Test if the average transformation required is very close to the
            # identity transformation and stop iteration if it is
            accuracy = np.mean(np.array([1, 0, 0, 0]) - np.mean(trans, axis=1))**2
            # If the accuracy starts to decrease then we have reached limit of precision
            # possible
            if accuracy > current_accuracy: converged = True
            else: current_accuracy = accuracy


        self.shapes = shapes