import numpy as np
import cv2

import math

def drange(start, stop, step):
  r = start
  while r < stop:
    yield r
    r += step

class utils:

    # determines waiting paramater in miliseconds for showing image, if 0 then user needs to close window manually
    imgWait = 1

    # determines scale factor for showing images
    imgScaleFactor = 0.7


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

        img_matrix = img_matrix.astype(np.uint8)

        img_matrix = cv2.resize(img_matrix, (0,0), fx=utils.imgScaleFactor, fy=utils.imgScaleFactor)

        cv2.imshow(title, img_matrix)

        cv2.waitKey(utils.imgWait)

    @staticmethod
    def drawShape(img,points):
        w = 2
        h = 2

        img = cv2.imread(img);

        for point in points:
            x = int(point.x)
            y = int(point.y)


            cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,255), 2)
        utils.displayImg('drawn shape',img)

    @staticmethod
    def produce_gradient_image(i, scale):


        i = cv2.cvtColor(i, cv2.COLOR_RGB2GRAY)

        height = i.shape[0]
        width = i.shape[1]

        black_image = np.zeros((height,width))

        start_x = 740
        end_x = 1250


        start_y = 1300
        end_y = 1750

        print(i.shape)

        i = i[start_x:end_x,start_y:end_y]


        i = i.astype(np.uint8)

        print(i.shape)


        #i = cv2.medianBlur(i,7)

        #i = cv2.bilateralFilter(i,15,80,80)



        i = cv2.fastNlMeansDenoising(i,None,20,21,5)



        i = np.uint8(i)
        i = cv2.Canny(i,20,30)

        i = cv2.Scharr(i,2,1,0)
        #i2 = cv2.Canny(i,20,30)

        #i = cv2.addWeighted(i1,0.5,i2,0.5,0)

        black_image[start_x:start_x+i.shape[0], start_y:start_y+i.shape[1]] = i

        i = black_image



        #i = cv2.fastNlMeansDenoising(i,None,30,15,10)

        utils.displayImg('black', i)

        #exit(1)

        #i = cv2.Sobel(i,cv2.CV_64F,1,0,ksize=3)

        #i = cv2.Laplacian(i,cv2.CV_64F)


        return i

    @staticmethod
    def getMaxAlongNormal2(point, points,counter, matrix):

        if((counter+1) % 40 == 0): # last point of a single tooth

                next_point = points[counter-39]
                previous_point = points[counter-1]
        elif (counter == 0): # counter at beggining
                previous_point = points[39]
                next_point = points[counter+1]
        else:
                previous_point = points[counter-1]
                next_point = points[counter+1]
        x = int(point.x)
        y = int(point.y)

        norm = utils.getNormalToPoint((x,y),previous_point,next_point,matrix)


        maxPoint = utils.scanNormal(norm,point,matrix)

        return maxPoint

    # scans the normal and finds the point with max intensity
    @staticmethod
    def scanNormal(norm, p,image):

        scale = 1

        height = image.shape[0]
        width = image.shape[1]
        # Find extremes of normal within the image
        # Test x first
        min_t = -p.x / norm[0]
        if p.y + min_t*norm[1] < 0:
          min_t = -p.y / norm[1]
        elif p.y + min_t*norm[1] > height:
          min_t = (height - p.y) / norm[1]

        # X first again
        max_t = (width - p.x) / norm[0]
        if p.y + max_t*norm[1] < 0:
          max_t = -p.y / norm[1]
        elif p.y + max_t*norm[1] > height:
          max_t = (height - p.y) / norm[1]

        # Swap round if max is actually larger...
        tmp = max_t
        max_t = max(min_t, max_t)
        min_t = min(min_t, tmp)

        # Get length of the normal within the image
        x1 = min(p.x+max_t*norm[0], p.x+min_t*norm[0])
        x2 = max(p.x+max_t*norm[0], p.x+min_t*norm[0])
        y1 = min(p.y+max_t*norm[1], p.y+min_t*norm[1])
        y2 = max(p.y+max_t*norm[1], p.y+min_t*norm[1])
        l = math.sqrt((x2-x1)**2 + (y2-y1)**2)

        max_pt = p
        max_edge = 0

        # Now check over the vector
        #v = min(max_t, -min_t)
        #for t in drange(min_t, max_t, (max_t-min_t)/l):
        search = 20+scale*10
        # Look 6 pixels to each side too
        for side in range(-6, 6):
          # Normal to normal...
          new_p = Point(p.x + side*-norm[1], p.y + side*norm[0])
          for t in drange(-search if -search > min_t else min_t, \
                           search if search < max_t else max_t , 1):

            x = int(norm[0]*t + new_p.x)
            y = int(norm[1]*t + new_p.y)
            if x < 0 or x > width or y < 0 or y > height:
              continue
    #        cv.Circle(img, (x, y), 3, (100,100,100))
            #print x, y, self.g_image.width, self.g_image.height
            if image[y-1, x-1] > max_edge:
              max_edge = image[y-1, x-1]
              max_pt = Point(new_p.x + t*norm[0], new_p.y + t*norm[1])

        return max_pt


    # gets a point with maximum intensity along a normal
    @staticmethod
    def getMaxAlongNormal(point, points,counter, matrix):
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

        # draw rectangle on a place of point
        #cv2.rectangle(matrix, (x,y), (x+w, y+h), (255,0,255), 2)

        a = sorted(pixels, key=lambda a_entry: a_entry[2])


        intensity = a[len(a)-1][2]
        x = int(a[len(a)-1][0])
        y = int(a[len(a)-1][1])

        #print(counter,intensity)


        # draw rectangle on a place of lowest intensity value
        #cv2.rectangle(matrix, (x,y), (x+w, y+h), (255,0,255), 2)

        point = Point(x,y)

        return point


    @staticmethod
    def getYLandmarks(points, image):

        counter = 0
        matrix = image

        landmarksY = []

        w = 2
        h = 2



        for point in points:

            maxPoint = utils.getMaxAlongNormal2(point, points,counter, matrix)

            x = int(maxPoint.x)
            y = int(maxPoint.y)

            # draw rectangle on a place of lowest intensity value
            cv2.rectangle(matrix, (x,y), (x+w, y+h), (255,255,255), 3)

            landmarksY.append(maxPoint)

            counter = counter +1

        # displays img with resulting points
        utils.displayImg('lines',matrix)

        return landmarksY


    @staticmethod
    def getNormalToPoint(start, previous_point, next_point, matrix):

        x_prev = int(previous_point.x)
        y_prev = int(previous_point.y)

        x_next = int(next_point.x)
        y_next = int(next_point.y)

        vectorX = x_prev - x_next
        vectorY = y_prev - y_next

        mag = math.sqrt(vectorX*vectorX + vectorY*vectorY);

        return (-vectorY/mag, vectorX/mag)


    # DEPRECATED
    @staticmethod
    def drawPerpendick(start,previous_point,next_point,matrix):

        lineSize = 30

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



    # gets accuracy of the current approximation versus the target shape
    @staticmethod
    def getLoss(shape, targetShape):
        if(len(shape.pts) == len(targetShape.pts)):

            totalDistance = 0
            for i in range(len(shape.pts)):
                point = shape.pts[i]
                targetPoint = targetShape.pts[i]

                # eucleidan distance
                distance = math.sqrt((point.x - targetPoint.x)**2 + (point.y - targetPoint.y)**2)

                totalDistance = totalDistance + distance

            totalDistance = totalDistance / len(shape.pts)
            return totalDistance
        else:
            print("points number does not match target shape")
            return None


class Point:
    def __init__(self,x,y):
        self.x = x;
        self.y = y;

    def get(self):
        return {'x':self.x, 'y':self.y}

    def dist(self, p):
        #Return the distance of this point to another point param p: The other point
        return math.sqrt((p.x - self.x)**2 + (p.y - self.y)**2)


