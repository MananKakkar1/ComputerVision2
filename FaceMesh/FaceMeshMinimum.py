import cv2 as cv
import time
import mediapipe as mp

video = cv.VideoCapture(0)
mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=2,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

cTime, pTime = 0, time.time()
while True:
    success, img = video.read()
    if not success:
        continue
    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    res = faceMesh.process(imgRGB)
    if res.multi_face_landmarks:
        for faceLms in res.multi_face_landmarks:
            mpDraw.draw_landmarks(img, faceLms, mpFaceMesh.FACEMESH_TESSELATION)
    cTime = time.time()
    fps = int(1 / max(cTime - pTime, 1e-6))
    pTime = cTime
    cv.putText(img, str(fps), (10, 70), cv.FONT_HERSHEY_COMPLEX, 3, (0, 0, 255), 3)
    cv.imshow('Video', img)
    if cv.waitKey(1) & 0xFF == ord('d'):
        break

video.release()
cv.destroyAllWindows()
