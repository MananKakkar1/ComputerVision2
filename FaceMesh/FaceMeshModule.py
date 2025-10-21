import cv2 as cv
import time
import mediapipe as mp


class FaceMeshTracker:
    def __init__(
        self,
        static_image_mode=False,
        maxFaces=2,
        refineLandmarks=True,
        detectionCon=0.5,
        trackCon=0.5,
    ):
        self.static_image_mode = static_image_mode
        self.maxFaces = maxFaces
        self.refineLandmarks = refineLandmarks
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.pTime = 0
        self.cTime = 0

        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(
            static_image_mode=self.static_image_mode,
            max_num_faces=self.maxFaces,
            refine_landmarks=self.refineLandmarks,
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

            self._findMesh(currImg, draw=True)

            self.cTime = time.time()
            fps = int(1 / max(self.cTime - self.pTime, 1e-6))
            self.pTime = self.cTime
            cv.putText(
                currImg,
                str(fps),
                (10, 70),
                cv.FONT_HERSHEY_COMPLEX,
                3,
                (0, 0, 255),
                3,
            )
            cv.imshow("Video", currImg)
            if cv.waitKey(1) & 0xFF == ord("d"):
                break

        videoInput.release()
        cv.destroyAllWindows()

    def _findMesh(self, img, draw=True):
        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        res = self.faceMesh.process(imgRGB)

        faces = []
        if res.multi_face_landmarks:
            for faceLms in res.multi_face_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(
                        img, faceLms, self.mpFaceMesh.FACEMESH_TESSELATION
                    )
                h, w, _ = img.shape
                pts = [
                    (int(lm.x * w), int(lm.y * h)) for lm in faceLms.landmark
                ]
                faces.append(pts)
        return faces

