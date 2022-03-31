import cv2
import os
import math
import dlib
import numpy as np
from skimage.feature import greycomatrix, greycoprops
from .pupil_detector import pupil_detection
from .tools import VectorTools, CvTools


class FaceDetector:

    def __init__(self, img_way):
        self.project_way = os.path.abspath('')
        # OpenCV detection
        self.eye_cascade = cv2.CascadeClassifier(self.project_way + '/data/classifiers/haarcascade_eye_tree_eyeglasses.xml')
        self.face_cascade = cv2.CascadeClassifier(self.project_way + '/data/classifiers/haarcascade_frontalface_default.xml')

        # face detection with dlib
        self.face_detector = dlib.get_frontal_face_detector()
        self.landmarks_detector = dlib.shape_predictor(
            self.project_way + '/data/classifiers/shape_predictor_68_face_landmarks.dat')

        self.vector_tools = VectorTools()

        # self.image_way = self.project_way + '/data/images/Alexandra_Daddario.jpeg'  # blue
        # self.image_way = self.project_way + '/data/images/Emma_Watson.jpg' #gray
        # self.image_way = self.project_way + '/data/images/Olivia_Wilde.jpeg' #green

        self.image_way = self.project_way + '/data/images/' + img_way

        self.face_img = []
        self.eyes_img = []

        self.iris_pointers = []
        self.radius_to_iris = 25
        self.eyes_color = None

        self.landmarks = []
        self.hair_landmark = None
        self.top_face_landmark = None
        self.forehead_landmark = None
        self.left_cheek = None
        self.right_cheek = None
        self.glcm_locations = []

        self.features = []
        self.normalized_features = []

    def get_features(self):
        img = cv2.imread(self.image_way)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_detector(gray, 1)

        for face in faces:
            landmarks = self.landmarks_detector(gray, face)
            landmarks = CvTools.shape_to_np(landmarks)
            for i, (x, y) in enumerate(landmarks):
                self.landmarks.append((x, y))

            (x, y, w, h) = CvTools.rect_to_bb(face)
            self.face_img.append((x, y, w, h))
            face_img = img[y: y + h, x: x + w]
            face_img_gray = gray[y:y + h, x:x + w]

            self.to_process_landmarks()
            self.get_glcm_features(gray)
            self.get_skin_colors_features(img)

            eyes = self.eye_cascade.detectMultiScale(face_img_gray, 1.1, 4)
            for i, (ex, ey, ew, eh) in enumerate(eyes):
                eye_img = face_img[ey:ey + eh, ex:ex + ew]

                pupil = pupil_detection(eye_img, int(ew/2), int(eh/2))
                pupil.start_detection()
                if pupil._pupil is not None:
                    self.get_iris_points(eye_img, pupil)
                else:
                    self.get_iris_points_2(eye_img, ew, eh)
                self.eyes_img.append((ex, ey, ew, eh))
            self.get_eyes_color_features()

    def find_eyes(self):
        img = cv2.imread(self.image_way)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_detector(gray, 1)

        for face in faces:
            landmarks = self.landmarks_detector(gray, face)
            landmarks = CvTools.shape_to_np(landmarks)
            for i, (x, y) in enumerate(landmarks):
                self.landmarks.append((x, y))
                cv2.circle(img, (x, y), 4, (255, 0, 0), -1)

            (x, y, w, h) = CvTools.rect_to_bb(face)
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
            self.face_img.append((x, y, w, h))
            face_img = img[y: y + h, x: x + w]
            face_img_gray = gray[y:y + h, x:x + w]

            self.to_process_landmarks()
            cv2.circle(img, (self.top_face_landmark[0], self.top_face_landmark[1]), 8, (191, 2, 138), -1)
            cv2.circle(img, (self.forehead_landmark[0], self.forehead_landmark[1]), 3, (191, 2, 138), -1)
            cv2.circle(img, (self.hair_landmark[0], self.hair_landmark[1]), 3, (191, 2, 138), -1)
            cv2.rectangle(img, (self.forehead_landmark[0] - 15, self.forehead_landmark[1] - 15),
                          (self.forehead_landmark[0] + 15, self.forehead_landmark[1] + 15), (0, 255, 0), 3)

            cv2.circle(img, (self.left_cheek[0], self.left_cheek[1]), 3, (191, 2, 138), -1)
            cv2.rectangle(img, (self.left_cheek[0] - 15, self.left_cheek[1] - 15),
                          (self.left_cheek[0] + 15, self.left_cheek[1] + 15), (0, 255, 0), 3)

            cv2.circle(img, (self.right_cheek[0], self.right_cheek[1]), 3, (191, 2, 138), -1)
            cv2.rectangle(img, (self.right_cheek[0] - 15, self.right_cheek[1] - 15),
                          (self.right_cheek[0] + 15, self.right_cheek[1] + 15), (0, 255, 0), 3)

            self.get_glcm_features(gray)
            self.get_skin_colors_features(img)

            eyes = self.eye_cascade.detectMultiScale(face_img_gray, 1.1, 4)
            for i, (ex, ey, ew, eh) in enumerate(eyes):
                cv2.rectangle(face_img, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 5)
                eye_img = face_img[ey:ey + eh, ex:ex + ew]

                pupil = pupil_detection(eye_img, int(ew/2), int(eh/2))
                pupil.start_detection()
                if pupil._pupil is not None:
                    self.get_iris_points(eye_img, pupil)
                else:
                    self.get_iris_points_2(eye_img, ew, eh)

                cv2.imshow("eye " + str(i), eye_img)
                self.eyes_img.append((ex, ey, ew, eh))
            self.get_eyes_color_features()
        cv2.imshow('img', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def to_process_landmarks(self):
        self.get_special_face_landmark()
        self.get_landmarks_features()
        self.to_normalize_landmarks_features()

    def get_special_face_landmark(self):
        difference = self.landmarks[8][0] - self.landmarks[27][0], self.landmarks[8][1] - self.landmarks[27][1]
        x = int(self.landmarks[8][0] - difference[0] * 1.85)
        y = int(self.landmarks[8][1] - difference[1] * 1.85)
        self.hair_landmark = (x, y)

        x = int(self.landmarks[8][0] - difference[0] * 1.6)
        y = int(self.landmarks[8][1] - difference[1] * 1.6)
        self.top_face_landmark = (x, y)

        x = int(self.landmarks[8][0] - difference[0] * 1.3)
        y = int(self.landmarks[8][1] - difference[1] * 1.3)
        self.forehead_landmark = (x, y)

        x = self.landmarks[29][0] + int(abs(self.landmarks[29][0] - self.landmarks[14][0]) / 2)
        y = self.landmarks[29][1] + int(abs(self.landmarks[29][1] - self.landmarks[14][1]) / 2)
        self.left_cheek = (x, y)

        x = self.landmarks[29][0] - int(abs(self.landmarks[29][0] - self.landmarks[2][0]) / 2)
        y = self.landmarks[29][1] + int(abs(self.landmarks[29][1] - self.landmarks[2][1]) / 2)
        self.right_cheek = (x, y)

    def get_landmarks_features(self):
        # Description of facial ratios used
        # 1. Eyes width / Distance between eyes
        self.get_feature(36, 39, 39, 42)

        # 2. Eyes width / Nose width
        self.get_feature(36, 39, 31, 35)

        # 3. Mouth width / Distance between eyes
        self.get_feature(48, 54, 39, 42)

        # 4. Distance between upper lip and jaw / Distance between eyes
        self.get_feature(51, 8, 39, 42)

        # 5. Distance between upper lip and jaw / Nose width
        self.get_feature(51, 8, 31, 35)

        # 6. Distance between eyes / Lip height
        self.get_feature(39, 42, 51, 57)

        # print("7. Nose width / Distance between eyes :")
        self.get_feature(31, 35, 39, 42)

        # print("8. Nose width / Upper lip height :")
        self.get_feature(31, 35, 51, 62)

        # 9. Distance between eyes / Distance between nose and mouth
        self.get_feature(39, 42, 33, 51)

        average_point = (int((self.landmarks[22][0] + self.landmarks[21][0]) / 2),
                         int((self.landmarks[22][1] + self.landmarks[21][1]) / 2))
        # 10.  Face top-eyebrows / Eyebrows-nose
        d1 = self.vector_tools.euclid_distance(self.top_face_landmark, average_point)
        d2 = self.vector_tools.euclid_distance(average_point, self.landmarks[33])
        self.features.append(d1 / d2)

        # 11. Eyebrows-nose / Nose-jaw
        d1 = self.vector_tools.euclid_distance(average_point, self.landmarks[33])
        d2 = self.vector_tools.euclid_distance(self.landmarks[33], self.landmarks[8])
        self.features.append(d1 / d2)

        # 12. Face top-eyebrows / Nose-Jaw
        d1 = self.vector_tools.euclid_distance(self.top_face_landmark, average_point)
        d2 = self.vector_tools.euclid_distance(self.landmarks[33], self.landmarks[8])
        self.features.append(d1 / d2)

        # 13. Distance between eyes / Nose width
        self.get_feature(39, 42, 31, 35)

        # 14. Face height / Face width
        d1 = self.vector_tools.euclid_distance(self.top_face_landmark, self.landmarks[8])
        d2 = self.vector_tools.euclid_distance(self.landmarks[15], self.landmarks[1])
        self.features.append(d1 / d2)

        # Description of Symmetry ratios used
        # 15. eyebrow length
        self.get_feature(21, 17, 22, 26)

        # 16. Lower lip length
        self.get_feature(48, 57, 54, 57)

        # 17. Upper eyebrow
        d1 = self.vector_tools.euclid_distance(self.landmarks[19], average_point)
        d2 = self.vector_tools.euclid_distance(self.landmarks[24], average_point)
        self.features.append(d1 / d2)

        # 18. Upper lip
        self.get_feature(48, 51, 54, 51)

        # 19. Nose
        self.get_feature(31, 33, 35, 33)

    def to_normalize_landmarks_features(self):
        mean = np.mean([self.features])
        std = np.std([self.features])

        z_values = [(item - mean) / std for item in self.features]
        for i, z in enumerate(z_values):
            ub = 1.618 if i < 14 else 1
            res = ((z - min(z_values)) / (max(z_values) - min(z_values))) * ub
            self.normalized_features.append(res)

    def get_feature(self, l1, l2, l3, l4):
        d1 = self.vector_tools.euclid_distance(self.landmarks[l1], self.landmarks[l2])
        d2 = self.vector_tools.euclid_distance(self.landmarks[l3], self.landmarks[l4])
        self.features.append(d1 / d2)

    def get_glcm_features(self, image):
        size = 15
        scratch_locations = [self.forehead_landmark, self.left_cheek, self.right_cheek]
        scratch_patches = []
        for loc in scratch_locations:
            scratch_patches.append(image[loc[1]:loc[1] + size,
                                   loc[0]:loc[0] + size])
        # face_img = img[y: y + h, x: x + w]
        dis_sim = []
        corr = []
        homogen = []
        energy = []
        contrast = []
        for patch in scratch_patches:
            glcm = greycomatrix(patch, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
            dis_sim.append(greycoprops(glcm, 'dissimilarity')[0, 0])
            corr.append(greycoprops(glcm, 'correlation')[0, 0])
            homogen.append(greycoprops(glcm, 'homogeneity')[0, 0])
            energy.append(greycoprops(glcm, 'energy')[0, 0])
            contrast.append(greycoprops(glcm, 'contrast')[0, 0])
        self.features += corr
        self.features += homogen
        self.features += energy
        self.features += contrast

        self.normalized_features += corr
        self.normalized_features += homogen
        self.normalized_features += energy
        self.normalized_features += contrast

    def get_skin_colors_features(self, img):
        skin_pointers = self.get_skin_pointers(img)
        average_colors = self.get_average_skin_color(skin_pointers)
        self.get_features_based_on_skin_color(average_colors)
        return img

    def get_skin_pointers(self, img):
        radius = 10
        skin_pointers = []
        for i, loc in enumerate([self.forehead_landmark, self.left_cheek, self.right_cheek, self.hair_landmark]):
            pointers = []
            for d in range(0, 360, 5):
                radian = (d * math.pi) / 180
                x, y = int(radius * math.cos(radian)), int(radius * math.sin(radian))
                pos_x, pos_y = loc[0] + x, loc[1] + y
                r = img[pos_y, pos_x, 2]
                g = img[pos_y, pos_x, 1]
                b = img[pos_y, pos_x, 0]
                pointers.append((int(r), int(g), int(b)))
            skin_pointers.append(pointers)
        return skin_pointers

    @staticmethod
    def get_average_skin_color(skin_pointers):
        res = []
        for loc in skin_pointers:
            pointer_r = [x[0] for x in loc]
            pointer_g = [x[1] for x in loc]
            pointer_b = [x[2] for x in loc]
            average_color = [int(np.average(pointer_r)), int(np.average(pointer_g)), int(np.average(pointer_b))]
            res.append(average_color)
        return res

    def get_features_based_on_skin_color(self, average_colors):
        for item in average_colors:
            self.features.append(item[0])
            self.features.append(item[1])
            self.features.append(item[2])

            self.normalized_features.append(item[0] / 255)
            self.normalized_features.append(item[1] / 255)
            self.normalized_features.append(item[2] / 255)

    # decision based on pupil detector
    def get_iris_points(self, eye_img, pupil):
        if pupil._pupil is not None:
            self.radius_to_iris = pupil._pupil[2] + 6
            pupil_x, pupil_y = pupil._pupil[0], pupil._pupil[1]

            for d in range(0, 360, 10):
                radian = (d * math.pi) / 180
                x, y = int(self.radius_to_iris * math.cos(radian)), int(self.radius_to_iris * math.sin(radian))
                pos_x, pos_y = pupil_x + x, pupil_y + y
                pos = (pos_x, pos_y)
                r = eye_img[pos_y, pos_x, 2]
                g = eye_img[pos_y, pos_x, 1]
                b = eye_img[pos_y, pos_x, 0]
                self.iris_pointers.append([int(r), int(g), int(b)])
                # if self.to_filter(int(r), int(g), int(b)) is not True:
                #     cv2.circle(eye_img, pos, 1, (0, 0, 255))

    # decision based on image's center
    def get_iris_points_2(self, eye_img, ew, eh):
        self.radius_to_iris = int(ew * 0.1)
        for d in range(0, 360, 10):
            radian = (d * math.pi) / 180
            x, y = int(self.radius_to_iris * math.cos(radian)), int(self.radius_to_iris * math.sin(radian))
            pos_x, pos_y = int(ew / 2) + x, int(eh / 2) + y
            pos = (pos_x, pos_y)
            r = eye_img[pos_y, pos_x, 2]
            g = eye_img[pos_y, pos_x, 1]
            b = eye_img[pos_y, pos_x, 0]
            self.iris_pointers.append([int(r), int(g), int(b)])
            # if self.to_filter(int(r), int(g), int(b)) is not True:
            #     cv2.circle(eye_img, pos, 1, (0, 0, 255))

    def get_eyes_color_features(self):
        self.eyes_color_filtration()
        self.get_features_based_on_eyes_color()

    def eyes_color_filtration(self):
        pointer_r = [x[0] for x in self.iris_pointers]
        pointer_g = [x[1] for x in self.iris_pointers]
        pointer_b = [x[2] for x in self.iris_pointers]
        pointer_r, pointer_g, pointer_b = self.filter_extreme_values(pointer_r, pointer_g, pointer_b)

        pointer_b = self.filter_by_median(pointer_b)
        pointer_g = self.filter_by_median(pointer_g)
        pointer_r = self.filter_by_median(pointer_r)
        self.eyes_color = [max(pointer_r), max(pointer_b), max(pointer_g)]
        print("Max color (RGB): " + str(self.eyes_color))

    @staticmethod
    def filter_extreme_values(a, b, c):
        for i in reversed(list(range(len(a)))):
            if a[i] == 255 and b[i] == 0 and c[i] == 0 or a[i] == 0 and b[i] == 255 and c[i] == 0 \
                    or a[i] == 0 and b[i] == 0 and c[i] == 255 or a[i] < 35 and b[i] < 35 and c[i] < 35:
                a.pop(i)
                b.pop(i)
                c.pop(i)
        return a, b, c

    @staticmethod
    def to_filter(a, b, c):
        if a == 255 and b == 0 and c == 0 or a == 0 and b == 255 and c == 0 \
                or a == 0 and b == 0 and c == 255 or a < 35 and b < 35 and c < 35:
            return True
        return False

    @staticmethod
    def filter_by_median(numbers):
        median = np.median(numbers)
        return [x for x in numbers if median * 0.8 <= x <= median * 1.2]

    def get_features_based_on_eyes_color(self):
        for color in self.eyes_color:
            self.features.append(color)
            self.normalized_features.append(color / 255)
