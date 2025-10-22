import cv2 as cv
import mediapipe as mp


class FaceTracker:
    def __init__(self, modelSelection=0, detectionCon=0.5):
        self.modelSelection = modelSelection
        self.detectionCon = detectionCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceDetection = mp.solutions.face_detection
        self.faceDetection = self.mpFaceDetection.FaceDetection(
            model_selection=self.modelSelection,
            min_detection_confidence=self.detectionCon,
        )
        self.results = None

    def findFaces(self, img, draw: bool = True):
        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        res = self.faceDetection.process(imgRGB)
        self.results = res
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
                    cv.putText(
                        img,
                        f"{int(score * 100)}%",
                        (x, max(20, y - 10)),
                        cv.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2,
                    )
                boxes.append(((x, y, bw, bh), score))
        return boxes

    def findPosition(self, img, faceNo: int = 0, draw: bool = False):
        bbox = None
        score = 0.0
        if self.results and self.results.detections:
            dets = self.results.detections
            if 0 <= faceNo < len(dets):
                detection = dets[faceNo]
                rel = detection.location_data.relative_bounding_box
                h, w, _ = img.shape
                x, y = int(rel.xmin * w), int(rel.ymin * h)
                bw, bh = int(rel.width * w), int(rel.height * h)
                score = float(detection.score[0]) if detection.score else 0.0
                bbox = (x, y, bw, bh)
                if draw and bbox is not None:
                    cv.rectangle(img, (x, y), (x + bw, y + bh), (0, 0, 255), 2)
                    cv.putText(
                        img,
                        f"{int(score * 100)}%",
                        (x, max(20, y - 10)),
                        cv.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2,
                    )
        return bbox, score
