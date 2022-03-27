import cv2
import os
import math
import webcolors
import numpy as np
from .pupil_detector import pupil_detection


class FaceDetector:

    def __init__(self):
        self.face_img = []
        self.eyes_img = []

        self.iris_pointers = []
        self.radius_to_iris = 25
        self.eyes_color = None

    def find_eyes(self):
        project_way = os.path.abspath('')

        face_cascade = cv2.CascadeClassifier(project_way + '/data/classifiers/haarcascade_frontalface_default.xml')
        eye_cascade = cv2.CascadeClassifier(project_way + '/data/classifiers/haarcascade_eye_tree_eyeglasses.xml')
        # image_way = project_way + '/data/images/Alexandra_Daddario.jpeg' #blue
        image_way = project_way + '/data/images/Emma_Watson.jpg' #gray
        # image_way = project_way + '/data/images/Olivia_Wilde.jpeg' #green


        img = cv2.imread(image_way)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 3)
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = img[y:y + h, x:x + w]

            self.face_img.append((x, y, w, h))
            face_img = img[y: y + h, x: x + w]

            eyes = eye_cascade.detectMultiScale(roi_gray)
            for i, (ex, ey, ew, eh) in enumerate(eyes):
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 5)

                eye_img = face_img[ey:ey + eh, ex:ex + ew]

                pupil = pupil_detection(eye_img, int(ew/2), int(eh/2))
                pupil.start_detection()
                # if pupil._pupil is not None:
                #     cv2.circle(eye_img, (pupil._pupil[0], pupil._pupil[1]), pupil._pupil[2], (255, 0, 0), 2)
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
            # cv2.circle(eye_img, pos, 1, (0, 0, 255))
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

    # not always works
    def get_iris(self, eye_img):
        Kernal = np.ones((3, 3), np.uint8)
        eye_gray = cv2.cvtColor(eye_img, cv2.COLOR_BGR2GRAY)

        ret, binary = cv2.threshold(eye_gray, 60, 255, cv2.THRESH_BINARY_INV)
        width, height = binary.shape
        binary = binary[int(0.1 * height):int(height - 0.1 * height), int(0.1 * width):int(width - 0.1 * width)]
        opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, Kernal)  ##Opening Morphology
        dilate = cv2.morphologyEx(opening, cv2.MORPH_DILATE, Kernal)  ##Dilate Morphology
        # cv2.imshow("cropped iris", dilate)

        contours, hierarchy = cv2.findContours(dilate, cv2.RETR_TREE,  ##Find contours
                                               cv2.CHAIN_APPROX_NONE)
        if len(contours) != 0:
            cnt = contours[0]
            M1 = cv2.moments(cnt)

            Cx1 = int(M1['m10'] / M1['m00'])  ##Find center of the contour
            Cy1 = int(M1['m01'] / M1['m00'])
            croppedImagePixelLength = int(0.4 * height)  ##Number of pixels we cropped from the image
            # center = (int(Cx1 + x + ex),
            #            int(Cy1 + y + ey + croppedImagePixelLength))  ##Center coordinates
            return Cx1, Cy1
