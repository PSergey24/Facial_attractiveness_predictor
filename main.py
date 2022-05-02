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
        self.normalized = []
        self.score = float(score.replace(',', '.'))


class Helper:

    def main(self):
        name = os.path.abspath('') + '/data/images/005.jpg'
        self.process_photo(name)

    @staticmethod
    def process_photo(name):
        face_detector = FaceDetector(name)
        face_detector.process_photo()

    def get_files_with_landmarks(self):
        way = os.path.abspath('') + '/data/images/notsure'
        files = self.get_files_list(way)
        for i, file in enumerate(files):
            if file != 'copy' and file.split('.')[1].lower() in ['jpg', 'webp', 'jpeg', 'png']:
                self.process_photo(way + '/' + file)
                print(i, i / len(files))

    @staticmethod
    def get_files_list(path):
        temp = map(lambda name: os.path.join(path, name), os.listdir(path))
        return [name.split('/')[-1] for name in temp]

    @staticmethod
    def rename_files():
        path = 'data/images/4/working_version'
        files = os.listdir(path)

        for index, file in enumerate(files):
            extension = file.split('.')[-1]
            name = ''.join(["default_" + "{0:0>4}".format(index), '.', extension])
            print(name)
            os.rename(os.path.join(path, file), os.path.join(path, name))

    def get_features(self):
        good_files = self.get_files_list('data/images/4/working_version')
        photos = self.get_scores(good_files)
        self.save_features_to_file(photos)

    @staticmethod
    def get_scores(files):
        with open('data/Face_score_new.csv', 'r', newline='') as csvfile:
            csv_reader = csv.reader(csvfile, delimiter=',')
            photos = [(row[0], row[1]) for i, row in enumerate(csv_reader) if i != 0 and row[0] in files]
        return photos

    def save_features_to_file(self, photos):
        list_features = []
        for i, photo in enumerate(photos):
            way, score = photo
            way = os.path.abspath('') + '/data/images/4/working_version/' + way
            features = self.get_feature_handler(way)
            if features:
                img_info = ImageFeature(way, features, score)
                list_features.append(img_info)
            print(f'Processed photo number {i}: {way}')

        with open('data/image_features.pkl', 'ab') as outp:
            pickle.dump(list_features, outp, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def get_feature_handler(way):
        face_detector = FaceDetector(way)
        try:
            face_detector.get_photo_features()
            return face_detector.features
        except Exception:
            print(f'Get features mistake: {way}')

    def to_normalize(self):
        images = self.read_features_from_file('data/image_features.pkl')
        images = [item for item in images if len(item.features) == 39]
        min_features = [10000] * 39
        max_features = [0] * 39

        features = []
        for item in images:
            features.append(item.features)
            for i, feature in enumerate(item.features):
                min_features[i] = min(min_features[i], feature)
                max_features[i] = max(max_features[i], feature)

        for item in images:
            for i, feature in enumerate(item.features):
                normalized = 2 * ((feature - min_features[i])/(max_features[i] - min_features[i])) - 1
                item.normalized.append(normalized)

        with open('data/image_features_normalized.pkl', 'ab') as outp:
            pickle.dump(images, outp, pickle.HIGHEST_PROTOCOL)

    def train_model(self):
        linear_model = LinearModel(39, 256, 1)
        features = self.read_features_from_file('data/image_features_normalized.pkl')

        train, test = self.get_data(features)
        trainer = Trainer(linear_model)
        trainer.start(train, test)

    @staticmethod
    def read_features_from_file(name):
        with open(name, 'rb') as inp:
            features = pickle.load(inp)
            return features

    @staticmethod
    def get_data(features):
        random.shuffle(features)
        train = features[:int(len(features) * 0.85)]
        test = features[int(len(features) * 0.85):]
        return train, test


if __name__ == '__main__':
    helper = Helper()
    helper.main()

    # helper.get_files_with_landmarks()
    # helper.rename_files()

    # helper.get_features()
    # helper.to_normalize()

    # helper.train_model()
