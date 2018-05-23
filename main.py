import glob
import math

import cv2
import numpy as np
from scipy.spatial import procrustes
from lib.procrustes import ProcusterAnalysis

from lib.Shape import Shape

from lib.image import Image

from lib.utils import utils

import random

class ActiveShape:


    def __init__(self):

        self.images = []

        self.loadImages()

        procrustes = ProcusterAnalysis(self.images)

        self.shapes = procrustes.shapes
        self.w = procrustes.w


        print('procrusted point',self.shapes[1].pts[0].y)


        # obtain shape matrix from points
        self.getShapesMatrix()

        (self.evals, self.evecs, self.mean, self.modes) = \
        self.__construct_model(self.shapes)

        print(np.sum(self.mean))



        # calculating pca
        #self.pca()

        # constructing starting shape points from mean shape vector
        self.shape = self.shapes[0]
        #self.shape.points = self.shape.vecToPoints(self.pca_model['mean'])

        #self.iterateModel()


    def generate_example(self, b):
        """ b is a vector of floats to apply to each mode of variation
        """
        # Need to make an array same length as mean to apply to eigen
        # vectors
        full_b = np.zeros(len(self.mean))
        for i in range(self.modes): full_b[i] = b[i]

        p = self.mean
        for i in range(self.modes): p = p + full_b[i]*self.evecs[:,i]

        # Construct a shape object
        return Shape.from_vector(p)

    def getShapesMatrix(self):

        # dimensions have to be N x P
        # N = number of images
        # P = number of teeth in an image * number of points per tooth * 2 (2 represent x and y values)
        matrix = np.empty((0,640),np.uint8)

        for shape in self.shapes:
            landmarksVector = []
            for point in shape.pts:
                landmarksVector.append(point.x)
                landmarksVector.append(point.y)


            matrix = np.vstack([matrix,landmarksVector])
        self.landmarksMatrix = matrix

    def __construct_model(self, shapes):
        """ Constructs the shape model
        """
        shape_vectors = np.array([s.get_vector() for s in self.shapes])
        mean = np.mean(shape_vectors, axis=0)

        # Move mean to the origin
        # FIXME Clean this up...
        mean = np.reshape(mean, (-1,2))
        min_x = min(mean[:,0])
        min_y = min(mean[:,1])

        #mean = np.array([pt - min(mean[:,i]) for i in [0,1] for pt in mean[:,i]])
        #mean = np.array([pt - min(mean[:,i]) for pt in mean for i in [0,1]])
        mean[:,0] = [x - min_x for x in mean[:,0]]
        mean[:,1] = [y - min_y for y in mean[:,1]]
        #max_x = max(mean[:,0])
        #max_y = max(mean[:,1])
        #mean[:,0] = [x/(2) for x in mean[:,0]]
        #mean[:,1] = [y/(3) for y in mean[:,1]]
        mean = mean.flatten()
        #print mean

        # Produce covariance matrix
        cov = np.cov(shape_vectors, rowvar=0)
        # Find eigenvalues/vectors of the covariance matrix
        evals, evecs = np.linalg.eig(cov)

        # Find number of modes required to describe the shape accurately
        t = 0
        for i in range(len(evals)):
          if sum(evals[:i]) / sum(evals) < 0.99:
            t = t + 1
          else: break
        print "Constructed model with %d modes of variation" % t
        return (evals[:t], evecs[:,:t], mean, t)


    def pca(self):

        X = self.landmarksMatrix.astype(np.float64)
        [n,d] = X.shape

        nb_components = n

        MU = X.mean(axis=0)
        for i in range(n):
            X[i,:] -= MU

        S = (np.dot(X, X.T) / float(n))

        eigenvalues, eigenvectors = np.linalg.eigh(S)

        s = np.where(eigenvalues < 0)
        eigenvalues[s] = eigenvalues[s] * -1.0
        eigenvectors[:,s] = eigenvectors[:,s] * -1.0

        indexes = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[indexes]
        eigenvectors = eigenvectors[:,indexes][:,0:nb_components]

        eigenvectors = np.dot(X.T, eigenvectors)

        for i in range(nb_components):
            eigenvectors[:,i] = self.normalize_vector(eigenvectors[:,i],eigenvalues[i])

        self.pca_model = {'mean': MU, 'eigenvalues':eigenvalues,'eigenvectors':eigenvectors}

    def normalize_vector(self,vector,eigenval):
        return vector;

    def doIteration2(self, image):



        Yshape = Shape([])

        print(self.shape.pts)
        Yshape.pts = utils.getYLandmarks(self.shape.pts, image)

        meanPoints = Shape.from_vector(self.mean)

        print(Yshape.pts)

        new_s = Yshape.align_to_shape(Shape.from_vector(self.mean), self.w)

        var = new_s.get_vector() - self.mean
        new = self.mean
        for i in range(len(self.evecs.T)):
            b = np.dot(self.evecs[:,i],var)
            max_b = 2*math.sqrt(self.evals[i])
            b = max(min(b, max_b), -max_b)
            new = new + self.evecs[:,i]*b

        self.shape = Shape.from_vector(new).align_to_shape(Yshape, self.w)

        print(self.shape.pts[0].y)

        #utils.drawShape(image,self.shape.pts)

    '''
    def doIteration(self, image):

        # get new shape based on previous iteration shape
        # starting shape is a mean shape
        Yshape = Shape([])
        Yshape.points = utils.getYLandmarks(self.shape.points, image)

        print(Shape.from_vector(self.mean))

        new_s = Yshape.align_to_shape(Shape.from_vector(self.mean), self.w)


        shape = Yshape.pointsToVec()

        mean_reshaped = np.reshape(self.pca_model['mean'],(640,1))
        shape_reshaped = np.reshape(shape,(640,1))

        print(mean_reshaped)


        var = shape_reshaped - mean_reshaped.flatten()


        new = self.pca_model['mean']
        evecs = self.pca_model['eigenvectors']
        evals = self.pca_model['eigenvalues']



        for i in range(len(evecs.T)):
            #print(evals[i])

            #print(var)
            #b = np.dot(evecs[:,i],var)

            b = np.sum(np.multiply(evecs[:,i].T,var))

            #print(evecs[:,i].sum(),var.sum(),b)

            max_b = 2*math.sqrt(evals[i])
            b = max(min(b, max_b), -max_b)

            #b = random.uniform(-0.7,0.7)
            #print(i,b)
            print(np.sum(evecs[:,i]*b))
            new = new + np.dot(evecs[:,i],b).transpose()


        print("printing final shape")
        self.shape.points = self.shape.vecToPoints(new)

        self.shape.drawShape()



        #self.shape = Shape.from_vector(new).align_to_shape(s, self.asm.w)

    '''
    def loadImages(self):
        images = []
        dir_images = "_Data/Radiographs/"

        files = glob.glob(dir_images+"*.tif")

        for file in files:
            img_matrix = cv2.imread(file)
            name = str(file).replace('_Data/Radiographs\\','')

            image = Image(name,img_matrix)
            images.append(image)


        self.images = images

model = ActiveShape()


for i in range(2):

    model.doIteration2('_Data/Radiographs/01.tif')

