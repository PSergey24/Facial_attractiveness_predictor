from modules import FaceDetector
from modules import LinearModel, Trainer


def main():
    # face_detector = FaceDetector('photo_01.jpg')
    # face_detector.find_eyes()

    photos = {'Alexandra_Daddario.jpeg': 6, 'Emma_Watson.jpg': 5, 'Olivia_Wilde.jpeg': 6,
              'Anne_H.jpeg': 0, 'Liz_1.jpeg': 0, 'Liz_2.jpeg': 0, 'photo_01.jpg': 0}
    linear_model = LinearModel(46, 256, 1)

    data = []
    data_test = []
    for way, score in photos.items():
        face_detector = FaceDetector(way)
        face_detector.get_features()
        if score == 0:
            data_test.append(face_detector.normalized_features)
        else:
            data.append(face_detector.normalized_features + [score])

    trainer = Trainer(linear_model)
    trainer.start(data, data_test)


if __name__ == '__main__':
    main()

