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
    imgWait = 0

    # determines scale factor for showing images
    imgScaleFactor = 0.5

    # displays and image with a scaling factor
    # title = title of the image
    # img_matrix = image matrix
    @staticmethod
    def displayImg(title, img_matrix):

        img_matrix = img_matrix.astype(np.uint8)

        img_matrix = cv2.resize(img_matrix, (0,0), fx=utils.imgScaleFactor, fy=utils.imgScaleFactor)

        cv2.imshow(title, img_matrix)

        cv2.waitKey(utils.imgWait)

    # draws a shape from its points
    @staticmethod
    def drawShape(title, img,points):
        w = 2
        h = 2

        if(isinstance(img,basestring)):

            img = cv2.imread(img);

        for point in points:
            x = int(point.x)
            y = int(point.y)


            cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,255), 2)
        utils.displayImg(title,img)

    @staticmethod
    def clahe(image, clipLimit=2.0, gridSize=(32,32)):
        cl = cv2.createCLAHE(clipLimit, gridSize)
        return cl.apply(image)

    @staticmethod
    def threshold(img, threshold):
        b = np.empty(img.shape)
        (h,w) = img.shape
        for i in range(h):
            for j in range(w):
                b[i,j] = 255 if img[i,j] > threshold else 0
        return b

    @staticmethod
    def applyFilters(i):

        i = utils.clahe(i)


        sobelx = cv2.Sobel(i, cv2.CV_64F, 1, 0, ksize=9)
        sobely = cv2.Sobel(i, cv2.CV_64F, 0, 1, ksize=9)
        i = np.sqrt(sobelx**2 + sobely**2)

        #i = utils.threshold(i,200000)

        #utils.displayImg('gradient image', i)


        return i


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

        i = i[start_x:end_x,start_y:end_y]

        i = utils.applyFilters(i)

        black_image[start_x:start_x+i.shape[0], start_y:start_y+i.shape[1]] = i

        i = black_image

        return i

    # gets max along a normal to the current point
    # point = actual point
    # points = all points in a shape
    # counter = counter of an actual point in all points
    # matrix = image pixel matrix
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
        x = int(point.x)
        y = int(point.y)


        # gets normal vector
        norm = utils.getNormalToPoint((x,y),previous_point,next_point,matrix)

        # obtains max point on the normal
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


    @staticmethod
    def getYLandmarks(points, image):

        counter = 0
        matrix = image

        landmarksY = []

        w = 2
        h = 2



        for point in points:

            maxPoint = utils.getMaxAlongNormal(point, points,counter, matrix)

            x = int(maxPoint.x)
            y = int(maxPoint.y)

            # draw rectangle on a place of lowest intensity value
            cv2.rectangle(matrix, (x,y), (x+w, y+h), (255,255,255), 3)

            landmarksY.append(maxPoint)

            counter = counter +1



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


