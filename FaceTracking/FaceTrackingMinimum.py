import cv2 as cv
import time
import mediapipe as mp

mpFaceDetection = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
faceDetection = mpFaceDetection.FaceDetection()
video = cv.VideoCapture(0)
pTime = time.time()
while True:
    success, img = video.read()
    if not success:
        continue
    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    res = faceDetection.process(imgRGB)
    if res.detections:
        for id, detection in enumerate(res.detections):
            mpDraw.draw_detection(img, detection)
            bBox = detection.location_data.relative_bounding_box
            height, width, c = img.shape
            x, y = int(bBox.xmin * width), int(bBox.ymin * height)
            w, h = int(bBox.width * width), int(bBox.height * height)
            cv.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
    cTime = time.time()
    fps = int(1 / max(cTime - pTime, 1e-6))
    pTime = cTime
    cv.putText(img, str(fps), (10, 70), cv.FONT_HERSHEY_COMPLEX, 3, (0, 0, 255), 3)
    cv.imshow('Video', img)
    if cv.waitKey(1) & 0xFF == ord('d'):
        break

video.release()
cv.destroyAllWindows()
