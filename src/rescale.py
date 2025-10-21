import cv2 as cv

# function to resize video or photo
def rescaleFrame(frame, scale=0.75):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimesions = (width, height)
    return cv.resize(frame, dimesions, interpolation=cv.INTER_AREA)

# function to change resolution of image or video
def changeResolution(width, height):
    catVid.set(3, width)
    catVid.set(4, height)

catVid = cv.VideoCapture(r'C:\Users\manan\ComputerVision2\Resources\Videos\kitten.mp4')
while True:
    isTrue, frame = catVid.read() # get frame by frame of video.
    frameResized = rescaleFrame(frame)
    cv.imshow('Cat Video', frame) # show the frame.
    cv.imshow('Resized Video', frameResized) # resized frame
    if cv.waitKey(20) & 0xFF==ord('d'): # if d is pressed, stop displaying the video.
        break
catVid.release() # destroy capture device 
cv.destroyAllWindows() # destroy current open windows