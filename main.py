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

        self.shapes = ProcusterAnalysis(self.images).shapes

        print(self.shapes[1].points[0].x)

        self.getLandmarksMatrix()



        self.pca()

        self.shape = Shape()
        self.shape.points = self.shape.vecToPoints(self.pca_model['mean'])

        #self.iterateModel()

        # doing procuster analysis
        # swithc with PCA when its working



    def getLandmarksMatrix(self):

        # dimensions have to be N x P
        # N = number of images
        # P = number of teeth in an image * number of points per tooth * 2 (2 represent x and y values)
        matrix = np.empty((0,640),np.uint8)

        for shape in self.shapes:
            landmarksVector = []
            for point in shape.points:
                landmarksVector.append(point.x)
                landmarksVector.append(point.y)


            matrix = np.vstack([matrix,landmarksVector])
        self.landmarksMatrix = matrix


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



    def doIteration(self):

        # get new shape based on previous iteration shape
        # starting shape is a mean shape
        Yshape = Shape([])
        Yshape.points = utils.getYLandmarks(self.shape.points)


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


for i in range(1):
    model.doIteration()

