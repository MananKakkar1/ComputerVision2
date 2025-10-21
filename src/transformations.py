import cv2 as cv
import numpy as np

img = cv.imread(r'C:\Users\manan\ComputerVision2\Resources\Photos\cat.jpg')

cv.imshow('Cat', img)

def translate(img, x, y):
    transMat = np.float32([[1, 0, x], [0, 1, y]])
    dimensions = (img.shape[1], img.shape[0]) # (width, height)
    return cv.warpAffine(img, transMat, dimensions)

# -x --> translates image left
# -y --> translates image up
# x --> translates image right
# y --> translates image down

translated = translate(img, -100, 100)
cv.imshow('Transformed', translated)


def rotate(img, angle, rotPoint=None):
    (height, width) = img.shape[:2]
    if rotPoint is None:
        rotPoint = (width // 2, height // 2)
    rotMat = cv.getRotationMatrix2D(rotPoint, angle, 1.0)
    dimensions = (width, height)    
    return cv.warpAffine(img, rotMat, dimensions)

rotated = rotate(img, 90)
cv.imshow('Rotated', rotated)

# flip an image
# second arg: -1, 0, 1
# - -1 implies flip image over x and y axis
# - 0 implies flip image over x axis
# - 1 implies fip image over y
flipped = cv.flip(img, -1)
cv.imshow('Flipped', flipped)
cv.waitKey(0)