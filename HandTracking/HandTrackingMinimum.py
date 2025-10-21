import cv2 as cv
import mediapipe as mp
import time

cap = cv.VideoCapture(0)
mpHands = mp.solutions.hands
hands = mp.solutions.hands.Hands()
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0

while True:
    success, img = cap.read()
    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    res = hands.process(imgRGB)
    if res.multi_hand_landmarks:
        for handLms in res.multi_hand_landmarks:
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
    cTime = time.time()
    fps=int(1/(cTime-pTime))
    pTime = cTime
    cv.putText(img, str(fps), (10,70), cv.FONT_HERSHEY_COMPLEX, 3, (255, 255, 0), 3)
    cv.imshow("Video", img)
    cv.waitKey(1)