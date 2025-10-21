import cv2 as cv
import time
import mediapipe as mp

class PoseTracker:
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.pTime = 0
        self.cTime = 0
        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(
            static_image_mode=self.mode,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            smooth_segmentation=True,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon,
        )
    
    def startTracking(self):
        videoInput = cv.VideoCapture(0)
        self.pTime = time.time()
        while True:
            success, currImg = videoInput.read()
            if not success:
                continue
            self._findPose(currImg, draw=True)
            self.cTime = time.time()
            fps = int(1 / max(self.cTime - self.pTime, 1e-6))
            self.pTime = self.cTime
            cv.putText(currImg, str(fps), (10, 70), cv.FONT_HERSHEY_COMPLEX, 3, (0, 0, 255), 3)
            cv.imshow('Video', currImg)
            if cv.waitKey(1) & 0xFF == ord('d'):
                break
        videoInput.release()
        cv.destroyAllWindows()
    
    def _findPose(self, img, draw=True):
        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        res = self.pose.process(imgRGB)
        if res.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, res.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
                for id, lm in enumerate(res.pose_landmarks.landmark):
                    height, width, c = img.shape
                    cx, cy = int(lm.x * width), int(lm.y * height)
                    # id represents the ID number of the hand tracking node
                    # Now underneath, you can do something with each specific
                    # id node.
                    cv.circle(img, (cx, cy), 5, (0, 0, 255), cv.FILLED)
