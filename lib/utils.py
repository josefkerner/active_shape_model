import numpy as np
import cv2
import math

class utils:
    @staticmethod
    def createLineIterator(P1, P2, img):
        """
        Produces and array that consists of the coordinates and intensities of each pixel in a line between two points

        Parameters:
            -P1: a numpy array that consists of the coordinate of the first point (x,y)
            -P2: a numpy array that consists of the coordinate of the second point (x,y)
            -img: the image being processed

        Returns:
            -it: a numpy array that consists of the coordinates and intensities of each pixel in the radii (shape: [numPixels, 3], row = [x,y,intensity])
        """
       #define local variables for readability
        imageH = img.shape[0]
        imageW = img.shape[1]
        P1X = P1[0]
        P1Y = P1[1]
        P2X = P2[0]
        P2Y = P2[1]

        #difference and absolute difference between points
        #used to calculate slope and relative location between points
        dX = P2X - P1X
        dY = P2Y - P1Y
        dXa = np.abs(dX)
        dYa = np.abs(dY)

        #predefine numpy array for output based on distance between points
        itbuffer = np.empty(shape=(np.maximum(dYa,dXa),3),dtype=np.float32)
        itbuffer.fill(np.nan)

   #Obtain coordinates along the line using a form of Bresenham's algorithm
        negY = P1Y > P2Y
        negX = P1X > P2X
        if P1X == P2X: #vertical line segment
           itbuffer[:,0] = P1X
           if negY:
               itbuffer[:,1] = np.arange(P1Y - 1,P1Y - dYa - 1,-1)
           else:
               itbuffer[:,1] = np.arange(P1Y+1,P1Y+dYa+1)
        elif P1Y == P2Y: #horizontal line segment
           itbuffer[:,1] = P1Y
           if negX:
               itbuffer[:,0] = np.arange(P1X-1,P1X-dXa-1,-1)
           else:
               itbuffer[:,0] = np.arange(P1X+1,P1X+dXa+1)
        else: #diagonal line segment
           steepSlope = dYa > dXa
           if steepSlope:
               slope = dX.astype(np.float32)/dY.astype(np.float32)
               if negY:
                   itbuffer[:,1] = np.arange(P1Y-1,P1Y-dYa-1,-1)
               else:
                   itbuffer[:,1] = np.arange(P1Y+1,P1Y+dYa+1)
               itbuffer[:,0] = (slope*(itbuffer[:,1]-P1Y)).astype(np.int) + P1X
           else:
               slope = dY.astype(np.float32)/dX.astype(np.float32)
               if negX:
                   itbuffer[:,0] = np.arange(P1X-1,P1X-dXa-1,-1)
               else:
                   itbuffer[:,0] = np.arange(P1X+1,P1X+dXa+1)
               itbuffer[:,1] = (slope*(itbuffer[:,0]-P1X)).astype(np.int) + P1Y

        #Remove points outside of image
        colX = itbuffer[:,0]
        colY = itbuffer[:,1]
        itbuffer = itbuffer[(colX >= 0) & (colY >=0) & (colX<imageW) & (colY<imageH)]

        #Get intensities from img ndarray


        itbuffer[:,2] = img[itbuffer[:,1].astype(np.uint),itbuffer[:,0].astype(np.uint)]

        return itbuffer

    @staticmethod
    def displayImg(title, img_matrix):

        scaleFactor = 0.5

        wait = 0
        img_matrix = img_matrix.astype(np.uint8)

        img_matrix = cv2.resize(img_matrix, (0,0), fx=scaleFactor, fy=scaleFactor)

        cv2.imshow(title, img_matrix)

        cv2.waitKey(wait)

    @staticmethod
    def drawShape(img,points):
        w = 2
        h = 2

        for point in points:
            x = int(point.x)
            y = int(point.y)


            cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,255), 2)
        utils.displayImg('drawn shape',img)

    @staticmethod
    def getYLandmarks(points):

        counter = 0
        matrix = cv2.imread('_Data/Radiographs/01.tif')
        matrix = cv2.cvtColor(matrix, cv2.COLOR_BGR2GRAY)

        landmarksY = []

        for point in points:

            if((counter+1) % 40 == 0): # last point of a single tooth

                next_point = points[counter-39]
                previous_point = points[counter-1]
            elif (counter == 0): # counter at beggining
                previous_point = points[39]
                next_point = points[counter+1]
            else:
                previous_point = points[counter-1]
                next_point = points[counter+1]

            w = 2
            h = 2
            x = int(point.x)
            y = int(point.y)

            x_start, y_start, x_end, y_end = utils.drawPerpendick((x,y),previous_point,next_point,matrix)

            point1 = np.array([x_start,y_start])
            point2 = np.array([x_end,y_end])

            pixels = utils.createLineIterator(point1,point2,matrix)
            #cv2.line(matrix,(x_start,y_start),(x_end,y_end),(255,255,0))
            cv2.rectangle(matrix, (x,y), (x+w, y+h), (255,0,255), 2)

            a = sorted(pixels, key=lambda a_entry: a_entry[2])

            intensity = a[0][2]
            x = int(a[0][0])
            y = int(a[0][1])

            pointDict = Point(x,y)

            cv2.rectangle(matrix, (x,y), (x+w, y+h), (0,0,255), 2)

            landmarksY.append(pointDict)

            counter = counter +1

        utils.displayImg('lines',matrix)

        return landmarksY

    @staticmethod
    def drawPerpendick(start,previous_point,next_point,matrix):

        lineSize = 50

        x = start[0]
        y = start[1]

        x_prev = int(previous_point.x)
        y_prev = int(previous_point.y)

        x_next = int(next_point.x)
        y_next = int(next_point.y)

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


class Point:
    def __init__(self,x,y):
        self.x = x;
        self.y = y;

    def get(self):
        return {'x':self.x, 'y':self.y}

    def dist(self, p):
        #Return the distance of this point to another point param p: The other point
        return math.sqrt((p.x - self.x)**2 + (p.y - self.y)**2)


