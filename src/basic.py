import cv2 as cv

img = cv.imread(r'C:\Users\manan\ComputerVision2\Resources\Photos\cat.jpg')
cv.imshow('Cat', img)

# Converting to greyscale
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray',  gray)

# Blur (removes noise in image) and increase blur by increasing numbers in tupe (x, x) where x is odd
blur = cv.GaussianBlur(img, (3, 3), cv.BORDER_DEFAULT)
cv.imshow('Blur', blur)

# Edge Cascade (pass in blur to decrease edges in image)
canny = cv.Canny(blur, 125, 175)
cv.imshow('Canny Edges', canny)

# Dilate an image
dilated = cv.dilate(canny, (3, 3), iterations=1)
cv.imshow('Dilated', dilated)

# Eroding an image
eroded = cv.erode(dilated, (7, 7), iterations=3)
cv.imshow('Eroded', eroded)

# Resizing an image (Use INTER_AREA for shrinking image. Use INTER_LINEAR for enlarging the image. Use INTER_CUBIC for enlarging the image (this option is much slower but yields higher quality image))
resized = cv.resize(img, (500, 500), interpolation=cv.INTER_AREA)
cv.imshow('Resized', resized)

cropped = img[50:200, 200:400]
cv.imshow('Cropped', cropped)
cv.waitKey(0)