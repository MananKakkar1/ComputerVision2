import cv2 as cv
import time
import mediapipe as mp


class FaceTracker:
    def __init__(self, modelSelection=0, detectionCon=0.5):
        self.modelSelection = modelSelection
        self.detectionCon = detectionCon

        self.pTime = 0
        self.cTime = 0

        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceDetection = mp.solutions.face_detection
        self.faceDetection = self.mpFaceDetection.FaceDetection(
            model_selection=self.modelSelection,
            min_detection_confidence=self.detectionCon,
        )

    def startTracking(self):
        videoInput = cv.VideoCapture(0)
        self.pTime = time.time()
        while True:
            success, currImg = videoInput.read()
            if not success:
                continue
            self._findFaces(currImg, draw=True)
            self.cTime = time.time()
            fps = int(1 / max(self.cTime - self.pTime, 1e-6))
            self.pTime = self.cTime
            cv.putText(currImg, str(fps), (10, 70), cv.FONT_HERSHEY_COMPLEX, 3, (0, 0, 255), 3)
            cv.imshow('Video', currImg)
            if cv.waitKey(1) & 0xFF == ord('d'):
                break
        videoInput.release()
        cv.destroyAllWindows()

    def _findFaces(self, img, draw=True):
        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        res = self.faceDetection.process(imgRGB)
        boxes = []
        if res.detections:
            for detection in res.detections:
                bbox = detection.location_data.relative_bounding_box
                h, w, _ = img.shape
                x, y = int(bbox.xmin * w), int(bbox.ymin * h)
                bw, bh = int(bbox.width * w), int(bbox.height * h)
                score = float(detection.score[0]) if detection.score else 0.0
                if draw:
                    self.mpDraw.draw_detection(img, detection)
                    cv.rectangle(img, (x, y), (x + bw, y + bh), (0, 0, 255), 2)
                    cv.putText(img, f"{int(score * 100)}%", (x, max(20, y - 10)),
                               cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                boxes.append(((x, y, bw, bh), score))
        return boxes

