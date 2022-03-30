import cv2
import os
import math
import webcolors
import dlib
import numpy as np
from skimage.feature import greycomatrix, greycoprops
from .pupil_detector import pupil_detection
from .tools import VectorTools, CvTools


class FaceDetector:

    def __init__(self):
        self.project_way = os.path.abspath('')
        # OpenCV detection
        self.eye_cascade = cv2.CascadeClassifier(self.project_way + '/data/classifiers/haarcascade_eye_tree_eyeglasses.xml')
        self.face_cascade = cv2.CascadeClassifier(self.project_way + '/data/classifiers/haarcascade_frontalface_default.xml')

        # face detection with dlib
        self.face_detector = dlib.get_frontal_face_detector()
        self.landmarks_detector = dlib.shape_predictor(
            self.project_way + '/data/classifiers/shape_predictor_68_face_landmarks.dat')

        self.image_way = self.project_way + '/data/images/Alexandra_Daddario.jpeg'  # blue
        self.image_way = self.project_way + '/data/images/Emma_Watson.jpg' #gray
        # self.image_way = self.project_way + '/data/images/Olivia_Wilde.jpeg' #green

        self.face_img = []
        self.eyes_img = []

        self.iris_pointers = []
        self.radius_to_iris = 25
        self.eyes_color = None

        self.landmarks = []
        self.top_face_landmark = None
        self.forehead_landmark = None
        self.left_cheek = None
        self.right_cheek = None
        self.glcm_locations = []

        self.features = []
        self.normalized_features = []

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

            self.landmarks_process()
            cv2.circle(img, (self.top_face_landmark[0], self.top_face_landmark[1]), 8, (191, 2, 138), -1)
            cv2.circle(img, (self.forehead_landmark[0], self.forehead_landmark[1]), 3, (191, 2, 138), -1)
            cv2.rectangle(img, (self.forehead_landmark[0] - 15, self.forehead_landmark[1] - 15),
                          (self.forehead_landmark[0] + 15, self.forehead_landmark[1] + 15), (0, 255, 0), 3)

            cv2.circle(img, (self.left_cheek[0], self.left_cheek[1]), 3, (191, 2, 138), -1)
            cv2.rectangle(img, (self.left_cheek[0] - 15, self.left_cheek[1] - 15),
                          (self.left_cheek[0] + 15, self.left_cheek[1] + 15), (0, 255, 0), 3)

            cv2.circle(img, (self.right_cheek[0], self.right_cheek[1]), 3, (191, 2, 138), -1)
            cv2.rectangle(img, (self.right_cheek[0] - 15, self.right_cheek[1] - 15),
                          (self.right_cheek[0] + 15, self.right_cheek[1] + 15), (0, 255, 0), 3)

            self.get_glcm_features(gray)
            self.get_skin_color(img)

            eyes = self.eye_cascade.detectMultiScale(face_img_gray, 1.1, 4)
            for i, (ex, ey, ew, eh) in enumerate(eyes):
                cv2.rectangle(face_img, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 5)
                eye_img = face_img[ey:ey + eh, ex:ex + ew]

                pupil = pupil_detection(eye_img, int(ew/2), int(eh/2))
                pupil.start_detection()
                if pupil._pupil is not None:
                    eye_img = self.get_iris_points(eye_img, pupil)
                else:
                    eye_img = self.get_iris_points_2(eye_img, ew, eh)

                cv2.imshow("eye " + str(i), eye_img)
                self.eyes_img.append((ex, ey, ew, eh))
        self.get_eyes_color()
        cv2.imshow('img', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def landmarks_process(self):
        self.get_special_face_landmark()
        self.get_features()
        self.get_normalized_features()

    def get_special_face_landmark(self):
        difference = self.landmarks[8][0] - self.landmarks[27][0], self.landmarks[8][1] - self.landmarks[27][1]
        x = int(self.landmarks[8][0] - difference[0] * 1.6)
        y = int(self.landmarks[8][1] - difference[1] * 1.6)
        self.top_face_landmark = (x, y)

        x = int(self.landmarks[8][0] - difference[0] * 1.3)
        y = int(self.landmarks[8][1] - difference[1] * 1.3)
        self.forehead_landmark = (x, y)

        x = self.landmarks[29][0] + int(abs(self.landmarks[29][0] - self.landmarks[15][0]) / 2)
        y = self.landmarks[29][1] + int(abs(self.landmarks[29][1] - self.landmarks[15][1]) / 2)
        self.left_cheek = (x, y)

        x = self.landmarks[29][0] - int(abs(self.landmarks[29][0] - self.landmarks[1][0]) / 2)
        y = self.landmarks[29][1] - int(abs(self.landmarks[29][1] - self.landmarks[1][1]) / 2)
        self.right_cheek = (x, y)

    def get_features(self):
        vTools = VectorTools()

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
        d1 = vTools.euclid_distance(self.top_face_landmark, average_point)
        d2 = vTools.euclid_distance(average_point, self.landmarks[33])
        self.features.append(d1 / d2)

        # 11. Eyebrows-nose / Nose-jaw
        d1 = vTools.euclid_distance(average_point, self.landmarks[33])
        d2 = vTools.euclid_distance(self.landmarks[33], self.landmarks[8])
        self.features.append(d1 / d2)

        # 12. Face top-eyebrows / Nose-Jaw
        d1 = vTools.euclid_distance(self.top_face_landmark, average_point)
        d2 = vTools.euclid_distance(self.landmarks[33], self.landmarks[8])
        self.features.append(d1 / d2)

        # 13. Distance between eyes / Nose width
        self.get_feature(39, 42, 31, 35)

        # 14. Face height / Face width
        d1 = vTools.euclid_distance(self.top_face_landmark, self.landmarks[8])
        d2 = vTools.euclid_distance(self.landmarks[15], self.landmarks[1])
        self.features.append(d1 / d2)

        # Description of Symmetry ratios used
        # 15. eyebrow length
        self.get_feature(21, 17, 22, 26)

        # 16. Lower lip length
        self.get_feature(48, 57, 54, 57)

        # 17. Upper eyebrow
        d1 = vTools.euclid_distance(self.landmarks[19], average_point)
        d2 = vTools.euclid_distance(self.landmarks[24], average_point)
        self.features.append(d1 / d2)

        # 18. Upper lip
        self.get_feature(48, 51, 54, 51)

        # 19. Nose
        self.get_feature(31, 33, 35, 33)

    def get_normalized_features(self):
        mean = np.mean([self.features])
        std = np.std([self.features])

        z_values = [(item - mean) / std for item in self.features]
        for i, z in enumerate(z_values):
            ub = 1.618 if i < 14 else 1
            res = ((z - min(z_values)) / (max(z_values) - min(z_values))) * ub
            self.normalized_features.append(res)

    def get_feature(self, l1, l2, l3, l4):
        vTools = VectorTools()
        d1 = vTools.euclid_distance(self.landmarks[l1], self.landmarks[l2])
        d2 = vTools.euclid_distance(self.landmarks[l3], self.landmarks[l4])
        self.features.append(d1 / d2)

    def get_glcm_features(self, image):
        size = 15
        scratch_locations = [self.forehead_landmark, self.left_cheek, self.right_cheek]
        scratch_patches = []
        for loc in scratch_locations:
            scratch_patches.append(image[loc[0]:loc[0] + size,
                                   loc[0]:loc[0] + size])

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

    def get_skin_color(self, img):
        skin_pointers = self.get_skin_pointers(img)
        return img

    def get_skin_pointers(self, img):
        radius = 15
        skin_pointers = []
        for i, loc in enumerate([self.forehead_landmark, self.left_cheek, self.right_cheek]):
            pointers = []
            for d in range(0, 360, 5):
                radian = (d * math.pi) / 180
                x, y = int(radius * math.cos(radian)), int(radius * math.sin(radian))
                pos_x, pos_y = loc[0] + x, loc[1] + y
                pixel = img[pos_x][pos_y]
                pointers.append([int(pixel[0]), int(pixel[1]), int(pixel[2])])
            average_color = np.mean(pointers, axis=0)
            skin_pointers.append((int(average_color[0]), int(average_color[1]), int(average_color[2])))
        return skin_pointers

    # decision based on pupil detector
    def get_iris_points(self, eye_img, pupil):
        if pupil._pupil is not None:
            self.radius_to_iris = pupil._pupil[2] + 6
            pupil_x, pupil_y = pupil._pupil[0], pupil._pupil[1]

            for d in range(0, 360, 5):
                radian = (d * math.pi) / 180
                x, y = int(self.radius_to_iris * math.cos(radian)), int(self.radius_to_iris * math.sin(radian))
                pos_x, pos_y = pupil_x + x, pupil_y + y

                for i in range(4):
                    for j in range(4):
                        pixel = eye_img[pos_x+i][pos_y+j]
                        pos = (pos_x+i, pos_y+j)
                        self.iris_pointers.append([int(pixel[0]), int(pixel[1]), int(pixel[2])])
                        cv2.circle(eye_img, pos, 1, (0, 0, 255))
        return eye_img

    # decision based on image's center
    def get_iris_points_2(self, eye_img, ew, eh):
        self.radius_to_iris = int(ew * 0.1)
        for d in range(0, 360, 5):
            radian = (d * math.pi) / 180
            x, y = int(self.radius_to_iris * math.cos(radian)), int(self.radius_to_iris * math.sin(radian))
            pos_x, pos_y = int(ew / 2) + x, int(eh / 2) + y
            pos = (pos_x, pos_y)
            pixel = eye_img[pos_x][pos_y]
            self.iris_pointers.append([pixel[0], pixel[1], pixel[2]])
            cv2.circle(eye_img, pos, 1, (0, 0, 255))
        return eye_img

    def get_eyes_color(self):
        pointer_b = [x[0] for x in self.iris_pointers]
        pointer_g = [x[1] for x in self.iris_pointers]
        pointer_r = [x[2] for x in self.iris_pointers]
        # pointer_r, pointer_g, pointer_b = self.filter_extreme_values(pointer_r, pointer_g, pointer_b)

        # pointer_b = self.filter_by_median(pointer_b)
        # pointer_g = self.filter_by_median(pointer_g)
        # pointer_r = self.filter_by_median(pointer_r)
        self.eyes_color = [int(np.average(pointer_r)), int(np.average(pointer_b)), int(np.average(pointer_g))]
        print("Average color (RGB): " + str(self.eyes_color) + " is " + str(self.find_color(self.eyes_color)))

    def find_color(self, requested_colour):  # finds the color name from RGB values
        min_colours = {}
        closest_name = ''
        for name, key in webcolors.CSS3_HEX_TO_NAMES.items():
            r_c, g_c, b_c = webcolors.hex_to_rgb(name)
            rd = (r_c - requested_colour[0]) ** 2
            gd = (g_c - requested_colour[1]) ** 2
            bd = (b_c - requested_colour[2]) ** 2
            min_colours[(rd + gd + bd)] = key
            closest_name = min_colours[min(min_colours.keys())]
        return closest_name

    @staticmethod
    def filter_extreme_values(a, b, c):
        for i in reversed(list(range(len(a)))):
            if a[i] == 255 and b[i] == 0 and c[i] == 0 or a[i] == 0 and b[i] == 255 and c[i] == 0 or a[i] == 0 and b[i] == 0 and c[i] == 255:
                a.pop(i)
                b.pop(i)
                c.pop(i)
        return a, b, c

    @staticmethod
    def filter_by_median(numbers):
        median = np.median(numbers)
        return [x for x in numbers if median * 0.6 <= x <= median * 1.4]

    def get_circle(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 20,
                                   param1=50, param2=30, minRadius=0, maxRadius=50)
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                cv2.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 2)
                cv2.circle(img, (i[0], i[1]), 2, (0, 0, 255), 3)
        return img
