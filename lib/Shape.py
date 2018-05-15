import cv2
import numpy as np
import math
from utils import utils

class Shape:
    def __init__(self):
        self.points = []
        self.Ypoints = []

    def vecToPoints(self, vec):
        points = []

        counter = 0
        for coord in vec:

            if(counter % 2 == 0):
                x = coord
            else:
                y = coord
                coords = {'x':x,'y':y}
                points.append(coords)
            counter = counter +1
        self.points = points

    def drawShape(self):

        test_images = ['_Data/Radiographs/01.tif']

        for test_image in test_images:
            matrix = cv2.imread(test_image)

            utils.drawShape(matrix,self.points)

    def pointsToVec(self):
        #vec = np.empty((640,))

        vec = np.array([])

        for point in self.Ypoints:
            point_arr = [point['x'],point['y']]
            vec = np.append(vec,point_arr)

        return vec

    def getYLandmarks(self):

        counter = 0
        matrix = cv2.imread('_Data/Radiographs/01.tif')
        matrix = cv2.cvtColor(matrix, cv2.COLOR_BGR2GRAY)

        landmarksY = []

        for point in self.points:

            if((counter+1) % 40 == 0): # last point of a single tooth

                next_point = self.points[counter-39]
                previous_point = self.points[counter-1]
            elif (counter == 0): # counter at beggining
                previous_point = self.points[39]
                next_point = self.points[counter+1]
            else:
                previous_point = self.points[counter-1]
                next_point = self.points[counter+1]

            w = 2
            h = 2
            x = int(point['x'])
            y = int(point['y'])

            x_start, y_start, x_end, y_end = self.drawPerpendick((x,y),previous_point,next_point,matrix)

            point1 = np.array([x_start,y_start])
            point2 = np.array([x_end,y_end])

            pixels = utils.createLineIterator(point1,point2,matrix)
            #cv2.line(matrix,(x_start,y_start),(x_end,y_end),(255,255,0))
            cv2.rectangle(matrix, (x,y), (x+w, y+h), (255,0,255), 2)

            a = sorted(pixels, key=lambda a_entry: a_entry[2])

            intensity = a[0][2]
            x = int(a[0][0])
            y = int(a[0][1])

            pointDict ={'x':x,'y':y}

            cv2.rectangle(matrix, (x,y), (x+w, y+h), (0,0,255), 2)

            landmarksY.append(pointDict)

            counter = counter +1

        self.Ypoints = landmarksY
        utils.displayImg('lines',matrix)

    def drawPerpendick(self, start,previous_point,next_point,matrix):

        lineSize = 50

        x = start[0]
        y = start[1]

        x_prev = int(previous_point['x'])
        y_prev = int(previous_point['y'])

        x_next = int(next_point['x'])
        y_next = int(next_point['y'])

        vectorX = x_prev - x_next
        vectorY = y_prev - y_next

        mag = math.sqrt(vectorX*vectorX + vectorY*vectorY);

        vectorX = vectorX / mag
        vectorY = vectorY / mag

        temp = vectorX;
        vectorX = -vectorY;
        vectorY = temp;

        length = lineSize*(-1)

        x_start = int(x + vectorX * length)
        y_start = int(y + vectorY * length)

        length = lineSize

        x_end = int(x + vectorX * length)
        y_end = int(y + vectorY * length)

        return x_start,y_start,x_end,y_end
