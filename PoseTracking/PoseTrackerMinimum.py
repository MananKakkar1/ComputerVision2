import cv2 as cv
import mediapipe as mp
import time

mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
pose = mpPose.Pose()
video = cv.VideoCapture(0)
pTime, cTime = 0, 0
while True:
    success, img = video.read()
    if not success:
        break
    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    res = pose.process(imgRGB)
    if res.pose_landmarks:
        mpDraw.draw_landmarks(img, res.pose_landmarks, mpPose.POSE_CONNECTIONS)
        for id, lm in enumerate(res.pose_landmarks):
            height, width, c = img.shape
            cx, cy = int(lm.x * width), int(lm.y * height)
            cv.circle(img, (cx, cy), 5, (0, 0, 255), cv.FILLED)
    cTime = time.time()
    fps = int(1 / (cTime - pTime))
    pTime = cTime
    cv.putText(img, str(fps), (10, 70), cv.FONT_HERSHEY_COMPLEX, 3, (0, 0, 255), 3)
    cv.imshow('Video', img)
    cv.waitKey(1)
