import cv2 as cv

def produce_gradient_image(i, scale=1):
    size = i.shape
    grey_image = cv.CreateImage(size, 8, 1)

    size = [s/scale for s in size]
    grey_image_small = cv.CreateImage(size, 8, 1)

    cv.CvtColor(i, grey_image, cv.CV_RGB2GRAY)

    df_dx = cv.CreateImage(cv.GetSize(i), cv.IPL_DEPTH_16S, 1)
    cv.Sobel( grey_image, df_dx, 1, 1)
    cv.Convert(df_dx, grey_image)
    cv.Resize(grey_image, grey_image_small)#, interpolation=cv.CV_INTER_NN)
    cv.Resize(grey_image_small, grey_image)#, interpolation=cv.CV_INTER_NN)
    return grey_image


img = '_Data/Radiographs/01.tif'

matrix = cv.imread(img)

produce_gradient_image(matrix)