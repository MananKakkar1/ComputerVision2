import cv2 as cv
import numpy as np
blank = np.zeros((500, 500, 3), dtype='uint8')
cv.imshow('Blank', blank)

blank[:] = 0,255,0 # change entire blank board to green
cv.imshow('Green', blank)

blank[200:300, 300:400] = 0,0,255
cv.imshow('Red Square', blank)

cv.rectangle(blank, (0, 0), (250, 250), (0, 255, 0), thickness=2)
cv.waitKey(0)
