import os
import csv
import pickle
import random
from modules import FaceDetector
from modules import LinearModel, Trainer


class ImageFeature:
    def __init__(self, name, features, score):
        self.name = name
        self.features = features
        self.score = float(score.replace(',', '.'))


def train_model():
    linear_model = LinearModel(46, 256, 1)

    # good_files = get_files_list('data/images/3')
    # photos = get_scores(good_files)
    # save_features_to_file(photos)

    features = read_features_from_file()

    train, test = get_data(features)
    trainer = Trainer(linear_model)
    trainer.start(train, test)


def save_features_to_file(photos):
    list_features = []
    for i, photo in enumerate(photos):
        way, score = photo
        features = get_feature_handler(way)
        if features:
            img_info = ImageFeature(way, features, score)
            list_features.append(img_info)
        print(f'Processed photo number {i}: {way}')

    with open('data/image_features.pkl', 'ab') as outp:
        pickle.dump(list_features, outp, pickle.HIGHEST_PROTOCOL)


def get_feature_handler(way):
    face_detector = FaceDetector(way)
    try:
        face_detector.get_photo_features()
        return face_detector.features
    except Exception:
        print(f'Get features mistake: {way}')


def read_features_from_file():
    with open('data/image_features.pkl', 'rb') as inp:
        features = pickle.load(inp)
        return features


def get_data(features):
    random.shuffle(features)
    train = features[:int(len(features) * 0.9)]
    test = features[int(len(features) * 0.9):]
    return train, test


def get_scores(files):
    with open('data/Face_scores.csv', 'r', newline='') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',')
        photos = [(row[0], row[3]) for i, row in enumerate(csv_reader) if i != 0 and row[0] in files]
    return photos


class Helper:

    def main(self):
        name = os.path.abspath('') + '/data/images/005.jpg'
        self.process_photo(name)
        # train_model()

    @staticmethod
    def process_photo(name):
        face_detector = FaceDetector(name)
        face_detector.process_photo()

    def get_files_with_landmarks(self):
        way = os.path.abspath('') + '/data/images/notsure'
        files = self.get_files_list(way)
        for i, file in enumerate(files):
            if file != 'copy' and file.split('.')[1] in ['jpg', 'webp', 'jpeg']:
                self.process_photo(way + '/' + file)
                print(i, i / len(files))

    @staticmethod
    def get_files_list(path):
        temp = map(lambda name: os.path.join(path, name), os.listdir(path))
        return [name.split('/')[-1] for name in temp]

    @staticmethod
    def rename_files():
        path = 'data/images/2'
        files = os.listdir(path)

        for index, file in enumerate(files):
            name = "CF_" + "{0:0>4}".format(index)
            os.rename(os.path.join(path, file), os.path.join(path, ''.join([name, '.jpg'])))


if __name__ == '__main__':
    helper = Helper()
    helper.main()

    # helper.get_files_with_landmarks()
