import cv2 as cv
import time
import mediapipe as mp

class HandTracker:
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.maxHands,
            model_complexity=1,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon,
        )
        self.mpDraw = mp.solutions.drawing_utils
        self.pTime = 0
        self.cTime = 0

    def startTracking(self):
        videoInput = cv.VideoCapture(0)
        self.pTime = time.time()
        while True:
            success, currImg = videoInput.read()
            if not success:
                continue
            self._findHands(currImg, draw=False)
            self.cTime = time.time()
            fps = int(1 / max(self.cTime - self.pTime, 1e-6))
            self.pTime = self.cTime
            cv.putText(currImg, str(fps), (10, 70), cv.FONT_HERSHEY_COMPLEX, 3, (0, 0, 255), 3)
            cv.imshow('Video', currImg)
            if cv.waitKey(1) & 0xFF == ord('d'):
                break
        videoInput.release()
        cv.destroyAllWindows()

    def _findHands(self, img, draw=True):
        if not draw:
            return
        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        res = self.hands.process(imgRGB)
        if res.multi_hand_landmarks and draw:
            for handLms in res.multi_hand_landmarks:
                self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
                for id, lm in enumerate(handLms.landmark):
                    height, width, c = img.shape
                    cx, cy = int(lm.x * width), int(lm.y * height)
                    # id represents the ID number of the hand tracking node
                    # Now underneath, you can do something with each specific
                    # id node.
                    cv.circle(img, (cx, cy), 15, (0, 0, 255), cv.FILLED)
