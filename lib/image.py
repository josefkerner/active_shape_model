import cv2
import glob
from utils import utils
from utils import Point

import numpy as np

from Shape import Shape

class Image:



    def __init__(self,name,img_matrix):
        self.name = name
        self.img_matrix = img_matrix
        self.teeth = []

        self.points = []

        self.teethAligned = []

        self.loadLandmarks()

        self.toPoints()

        self.pts = self.points

        #self.showLandmarks()

    def toPoints(self):

        for tooth in self.teeth:


            for landmark in tooth['landmarks']:

                point = Point(int(landmark['x']), int(landmark['y']))
                self.points.append(point)
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

        utils.displayImg('points',matrix)

