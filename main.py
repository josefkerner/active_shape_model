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

    TESTING_MODE = 0
    TESTING_IMAGE_INDEX = 6


    def __init__(self):

        self.images = []

        self.loadImages()



        procrustes = ProcusterAnalysis(self.images)

        self.shapes = procrustes.shapes

        self.leaveOneOut(ActiveShape.TESTING_IMAGE_INDEX)
        self.w = procrustes.w

        self.gradientImage = None



        # computing PCA model
        (self.evals, self.evecs, self.mean, self.modes) = self.__construct_model(self.shapes)

        #PCAvisualize((self.evals, self.evecs, self.mean))


        # constructing starting shape points from mean shape vector
        self.shape = Shape.from_vector(self.mean)

        if(ActiveShape.TESTING_MODE == 1 ):
            utils.drawShape('mean shape', '_Data/Radiographs/01.tif', self.shape.pts);


    # used for training and testing
    # leaves selected shape from the training data
    # index = image number
    def leaveOneOut(self, index):
        # obtaining target shape for evaluation
        self.targetShape = Shape([])
        self.targetShape.pts = self.images[index-1].points

        self.shapes.pop(index-1)


    # constructs model
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


    # does iteration on an image
    def doIteration(self, image):

        Yshape = Shape([])

        imageMatrix = cv2.imread(image)

        if(self.gradientImage == None):

            # performed only in first iteration
            self.shape.placeShape(imageMatrix)
            self.gradientImage = utils.produce_gradient_image(imageMatrix,1)

        gradintImg = self.gradientImage.copy()

        Yshape.pts = utils.getYLandmarks(self.shape.pts, gradintImg)

        if(ActiveShape.TESTING_MODE == 1):
            utils.drawShape('Y points',image,Yshape.pts)

        meanPoints = Shape.from_vector(self.mean)

        new_s = Yshape.align_to_shape(Shape.from_vector(self.mean), self.w)

        var = new_s.get_vector() - self.mean
        new = self.mean

        for i in range(len(self.evecs.T)):

            b = np.dot(self.evecs[:,i],var)
            max_b = 3*math.sqrt(self.evals[i])
            b = max(min(b, max_b), -max_b)

            new = new + self.evecs[:,i]*b

        self.shape = Shape.from_vector(new).align_to_shape(Yshape, self.w)




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

    def showCurrentShape(self, image):
        utils.drawShape('new shape',image,self.shape.pts)

    def extractShape(self,imgPath):
        img = cv2.imread(imgPath)
        mask = np.zeros((img.shape[0], img.shape[1]))

        coordinates = []
        counter = 1
        for point in self.shape.pts:
            coord = [int(point.x), int(point.y)]
            coordinates.append(coord)

            if(counter % 40 == 0):
                # new tooth starting
                # draw current tooth
                coordinates = np.asarray(coordinates)
                cv2.fillConvexPoly(mask, coordinates, 1)
                coordinates = []


            counter = counter +1

        mask = mask.astype(np.bool)
        out = np.zeros_like(img)
        out[mask] = img[mask]

        utils.displayImg('image',out)



model = ActiveShape()



def iterateOnImage(img, iterations):


    previousAcc = 200
    for i in range(iterations):

        acc = model.doIteration(img)
        if (acc > previousAcc):
            break



        print(acc)
        previousAcc = acc

    model.showCurrentShape(img)
    model.extractShape(img)

iterateOnImage('_Data/Radiographs/06.tif',20)

