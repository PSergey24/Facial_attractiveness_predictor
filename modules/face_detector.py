import cv2
import os


class FaceDetector:

    def __init__(self):
        self.face = []
        self.eyes = []

    def find_eyes(self):
        project_way = os.path.abspath('')

        face_cascade = cv2.CascadeClassifier(project_way + '/data/classifiers/haarcascade_frontalface_default.xml')
        eye_cascade = cv2.CascadeClassifier(project_way + '/data/classifiers/haarcascade_eye_tree_eyeglasses.xml')
        image_way = project_way + '/data/images/Emma_Watson.jpg'

        img = cv2.imread(image_way)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 3)
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = img[y:y + h, x:x + w]

            self.face.append((x, y, w, h))
            eyes = eye_cascade.detectMultiScale(roi_gray)
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 5)
                self.eyes.append((ex, ey, ew, eh))

        cv2.imshow('img', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()