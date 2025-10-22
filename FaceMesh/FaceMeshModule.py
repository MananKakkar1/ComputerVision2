import cv2 as cv
import mediapipe as mp


class FaceMeshTracker:
    def __init__(
        self,
        static_image_mode: bool = False,
        maxFaces: int = 2,
        refineLandmarks: bool = True,
        detectionCon: float = 0.5,
        trackCon: float = 0.5,
    ) -> None:
        self.static_image_mode = static_image_mode
        self.maxFaces = maxFaces
        self.refineLandmarks = refineLandmarks
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(
            static_image_mode=self.static_image_mode,
            max_num_faces=self.maxFaces,
            refine_landmarks=self.refineLandmarks,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon,
        )
        self.results = None

    def findMesh(self, img, draw: bool = True):
        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        self.results = self.faceMesh.process(imgRGB)
        if self.results and self.results.multi_face_landmarks:
            for faceLms in self.results.multi_face_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(
                        img, faceLms, self.mpFaceMesh.FACEMESH_TESSELATION
                    )
        return img

    def findPosition(self, img, faceNo: int = 0, draw: bool = True):
        lmList = []
        if self.results and self.results.multi_face_landmarks:
            faces = self.results.multi_face_landmarks
            if 0 <= faceNo < len(faces):
                myFace = faces[faceNo]
                h, w, _ = img.shape
                for id, lm in enumerate(myFace.landmark):
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lmList.append([id, cx, cy])
                    if draw:
                        cv.circle(img, (cx, cy), 2, (0, 255, 0), cv.FILLED)
        return lmList
