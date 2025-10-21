import cv2 as cv
import numpy as np


img = cv.imread(r'C:\Users\manan\ComputerVision2\Resources\Photos\cat.jpg')
cv.imshow('Cat', img)

blank = np.zeros(img.shape, dtype="uint8")

# countours are boundaries of an object, the line or curve that joins the continuous 
# points along the boundary of objects. (are edges for the most part (not mathematically tho))

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('gray', gray)

blur = cv.blur(gray, (5, 5), cv.BORDER_DEFAULT)
cv.imshow('Blur', blur)
# canny used to grab edges of image 
canny = cv.Canny(blur, 125, 175)
cv.imshow('Canny Img', canny)

# ret, thresh = cv.threshold(gray, 125, 255, cv.THRESH_BINARY) # to binarize the current image

# RETR_TREE gives all heirarchical contours
# RETR_EXTERNAL gives all external contours
# RETR_LIST gives ALL contours in the image
# CHAIN_APPROX_NONE returns all contours
# CHAIN_APPROX_SIMPLE returns compresses redundant contours and gives the simplest ones


# contours is the list of all coordinates of contours in the image
# heirarchies gives a list of heirarchical images such as square in a circle, etc
contours, heirarchies = cv.findContours(canny, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
print(f'{len(contours)} contour(s) found!')

cv.drawContours(blank, contours, -1, (0,0,255), 2)
cv.imshow('Contours Drawn', blank)
cv.waitKey(0)
cv.destroyAllWindows()