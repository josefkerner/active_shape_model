import glob
import math

import cv2
import numpy as np
from scipy.spatial import procrustes
from lib.procrustes import ProcusterAnalysis

from lib.Shape import Shape

from lib.image import Image

from lib.utils import utils


from lib.PCAvisualize import PCAvisualize
import random

class ActiveShape:


    def __init__(self):

        self.images = []

        self.loadImages()

        procrustes = ProcusterAnalysis(self.images)

        self.shapes = procrustes.shapes

        # obtaining target shape for evaluation on the first model
        self.targetShape = Shape([])
        self.targetShape.pts = self.images[0].points


        self.shapes = self.shapes[1:]
        self.w = procrustes.w

        self.gradientImage = None


        # obtain shape matrix from points
        self.getShapesMatrix()

        # computing PCA model
        (self.evals, self.evecs, self.mean, self.modes) = self.__construct_model(self.shapes)

        PCAvisualize((self.evals, self.evecs, self.mean))


        # constructing starting shape points from mean shape vector
        self.shape = Shape.from_vector(self.mean)
        self.shape.placeModel()

        #utils.drawShape('_Data/Radiographs/03.tif',self.shape.pts)

        #self.shape.points = self.shape.vecToPoints(self.pca_model['mean'])

        #self.iterateModel()


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
        """ Constructs the shape model """
        shape_vectors = np.array([s.get_vector() for s in self.shapes])
        mean = np.mean(shape_vectors, axis=0)


        mean = np.reshape(mean, (-1,2))
        min_x = min(mean[:,0])
        min_y = min(mean[:,1])

        mean[:,0] = [x - min_x for x in mean[:,0]]
        mean[:,1] = [y - min_y for y in mean[:,1]]

        mean = mean.flatten()

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



    def doIteration(self, image):

        Yshape = Shape([])

        imageMatrix = cv2.imread(image)

        if(self.gradientImage == None):

            self.gradientImage = utils.produce_gradient_image(imageMatrix,1)

        gradintImg = self.gradientImage.copy()


        Yshape.pts = utils.getYLandmarks(self.shape.pts, gradintImg)

        meanPoints = Shape.from_vector(self.mean)

        new_s = Yshape.align_to_shape(Shape.from_vector(self.mean), self.w)

        var = new_s.get_vector() - self.mean
        new = self.mean

        #b = [700,-147,1,1,1,-47,-49,1,1,-10,1]
        for i in range(len(self.evecs.T)):

            b = np.dot(self.evecs[:,i],var)
            max_b = 3*math.sqrt(self.evals[i])
            b = max(min(b, max_b), -max_b)

            #print("b vector is:",b)
            new = new + self.evecs[:,i]*b
            #new = new + self.evecs[:,i]*b[i]

        self.shape = Shape.from_vector(new).align_to_shape(Yshape, self.w)


        utils.drawShape(image,self.shape.pts)

        return utils.getLoss(self.shape, self.targetShape)

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



def iterateOnImage(img, iterations):


    for i in range(iterations):

        acc = model.doIteration(img)

        print(acc)

iterateOnImage('_Data/Radiographs/01.tif',150)

