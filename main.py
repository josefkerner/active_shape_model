import glob
import math

import cv2
import numpy as np
from scipy.spatial import procrustes

from lib.Shape import Shape

import random


class PCAvisualize():
    def __init__(self):
        pass

class ProcusterAnalysis:
    # returns new images array with landmarks scaled, aligned and transformed
    def __init__(self,images):
        self.images = images
        self.doProcrustes()

    def doProcrustes(self):
        original_landmarks = self.getTeethMatrix(self.images[0].teeth)
        current_landmarks = self.getTeethMatrix(self.images[1].teeth)

        mtx1, mtx2, disparity = procrustes(original_landmarks, current_landmarks)

        print(mtx2.shape)



        print(disparity)
    def getTeethMatrix(self,teeth):
        teethMatrix = np.empty((0,80),np.uint8)

        for tooth in teeth:
            landmarksVector = []
            for landmark in tooth['landmarks']:

                    landmarksVector.append(landmark['x'])
                    landmarksVector.append(landmark['y'])
            teethMatrix = np.vstack([teethMatrix,landmarksVector])
        return teethMatrix

class Image:



    def __init__(self,name,img_matrix):
        self.name = name
        self.img_matrix = img_matrix
        self.teeth = []

        self.teethAligned = []

        self.loadLandmarks()

        self.showLandmarks()
    def loadLandmarks(self):
        dir_landmarks = '_Data/Landmarks/original'
        id = int(self.name.replace('.tif',''))

        files = glob.glob(dir_landmarks+"/landmarks"+str(id)+"-*.txt")


        # path = single tooth
        # files =  all teeth in an image
        for path in files:
            teethID = 1
            landmarks = []


            file = open(path,'r')
            counter = 0
            for row in file:
                row = str(row).rstrip()

                if(counter % 2 == 0):
                    x = row
                else:
                    y = row
                    coords = {'x':int(float(x)), 'y':int(float(y))}

                    landmarks.append(coords)
                counter = counter+1

            tooth = {'teethID': teethID,'landmarks':landmarks}
            self.teeth.append(tooth)


    def showLandmarks(self):
        w = 2
        h = 2
        matrix = self.img_matrix

        for tooth in self.teeth:
            for landmark in tooth['landmarks']:
                x = int(landmark['x'])
                y = int(landmark['y'])
                cv2.rectangle(matrix, (x,y), (x+w, y+h), (255,0,255), 2)

        #utils.displayImg('points',matrix)






class ActiveShape:


    def __init__(self):

        self.images = []

        self.loadImages()
        self.getLandmarksMatrix()

        self.pca()

        self.shape = Shape()
        self.shape.vecToPoints(self.pca_model['mean'])

        #self.iterateModel()

        # doing procuster analysis
        # swithc with PCA when its working

        self.images = ProcusterAnalysis(self.images)

    def getLandmarksMatrix(self):

        # dimensions have to be N x P
        # N = number of images
        # P = number of teeth in an image * number of points per tooth * 2 (2 represent x and y values)
        matrix = np.empty((0,640),np.uint8)

        for image in self.images:
            landmarksVector = []
            for tooth in image.teeth:
                for landmark in tooth['landmarks']:

                    landmarksVector.append(landmark['x'])
                    landmarksVector.append(landmark['y'])
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
        return vector / eigenval



    def doIteration(self):

        # get new shape based on previous iteration shape
        # starting shape is a mean shape
        self.shape.getYLandmarks()

        shape = self.shape.pointsToVec()

        mean_reshaped = np.reshape(self.pca_model['mean'],(640,1))
        shape_reshaped = np.reshape(shape,(640,1))

        print(mean_reshaped.shape)

        mtx1, mtx2, disparity = procrustes(mean_reshaped,shape_reshaped)

        shape = mtx2*shape_reshaped
        mean_procrusted = mean_reshaped*mtx1

        var = shape.flatten() - mean_procrusted.flatten()


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
        self.shape.vecToPoints(new)

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

for i in range(5):
    model.doIteration()

