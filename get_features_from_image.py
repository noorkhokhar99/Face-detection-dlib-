import dlib # face processing library Dlib

import numpy as np # data processing library numpy
import cv2 # Image Processing Library OpenCv

# Dlib Positive Face Detection
detector = dlib.get_frontal_face_detector()
# Dlib Face Feature Point Prediction
predictor = dlib.shape_predictor('data/dlib/shape_predictor_68_face_landmarks.dat')

img_rd = cv2.imread("data/samples/face_0.jpeg")

img_gray = cv2.cvtColor(img_rd, cv2.COLOR_RGB2GRAY)

faces = detector(img_gray, 0)

font = cv2.FONT_HERSHEY_SIMPLEX


if len(faces) != 0:
    #Detecting a face
    for i in range(len(faces)):
        landmarks = np.matrix([[p.x, p.y] for p in predictor(img_rd, faces[i]).parts()])
        for idx, point in enumerate(landmarks):
            # 68 point coordinates
            pos = (point[0, 0], point[0, 1])

            # Use cv2.circle to draw a circle for each feature point, a total of 68
            cv2.circle(img_rd, pos, 2, color=(139, 0, 0))
            # Write cv2.putText number 1-68
            cv2.putText(img_rd, str(idx + 1), pos, font, 0.2, (187, 255, 255), 1, cv2.LINE_AA)

    cv2.putText(img_rd, "faces: " + str(len(faces)), (20, 50), font, 1, (0, 0, 0), 1, cv2.LINE_AA)
else:
    # No face detected
    cv2.putText(img_rd, "no face", (20, 50), font, 1, (0, 0, 0), 1, cv2.LINE_AA)

# Parameter takes 0 You can drag the zoom window to 1
# cv2.namedWindow("image", 0)
cv2.namedWindow("image", 1)

cv2.imshow("image", img_rd)
cv2.waitKey(0)