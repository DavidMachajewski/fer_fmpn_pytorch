#
# FER2013
#
# classes
#
# anger, disgust, fear, happy, sad, surprise, neutral
# emotions = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}
#
#
# Preprocess FER2013
#
# 0. Create FER dataset folder if not existent
# 1. Create 10 split of FER2013
#
#
#
import numpy as np
import pandas as pd
from args2 import Setup
from sklearn.model_selection import KFold
from sklearn.utils import shuffle


class FERUtils():
    def __init__(self, args):
        self.dataset = self.__load_csv__(args)
        self.train = None
        self.test = None
        self.val = None

    def __load_csv__(self, args):
        dataset = pd.read_csv(args.fer_images)
        print(dataset.values.shape)
        print(dataset.head())
        return dataset

    def show_sample(self):
        pass


class AffectNetUtils():
    """
    0: neutral
    1: happiness
    2: sadness
    3: surprise
    4: fear
    5: disgust
    6: anger
    7: contempt
    8: None
    9: Uncertain
    10: No-Face
    """
    def __init__(self, args):
        self.args = args
        self.dataset = self.__load_csv__(args)
        self.train_dfs, self.test_dfs = [], []

    def __load_csv__(self, args):
        dataset = pd.read_csv(args.affectnet_manual)
        return dataset

    def __rd_sample__(self,
                      emotions=[1, 2, 3, 4, 5, 6, 7],
                      n_samples=None):
        """Randomly sample 3500 images per class out of the
        manual annotated AffectNet dataset (We use the seven basic emotions)
        :return:
        """
        if n_samples is None:
            n_samples = {
                '1': 5000,
                '2': 5000,
                '3': 5000,
                '4': 5000,
                '5': 3750,
                '6': 5000,
                '7': 3750
            }
        train_splits = [[] for i in range(10)]
        train_label_splits = [[] for i in range(10)]
        test_splits = [[] for i in range(10)]
        test_label_splits = [[] for i in range(10)]
        for idx in range(len(emotions)):
            # get all files with emotion type idx
            tmp = self.dataset.loc[self.dataset['expression'] == emotions[idx]]
            tmp = tmp[['subDirectory_filePath', 'expression']]
            # get n_samples randomly
            tmp = tmp.sample(n = n_samples[str(emotions[idx])])
            tmp_train = tmp['subDirectory_filePath'].to_numpy()
            tmp_label = tmp['expression'].to_numpy()

            # create 10 fold crop of all files of type emotions[idx]
            kf = KFold(n_splits=10, shuffle=True)
            for split_id, (train_idx, test_idx) in enumerate(kf.split(tmp_train)):
                # print("train_idx: {0}, \n test_idx: {1}".format(train_idx, test_idx))

                train_files, test_files = tmp_train[train_idx], tmp_train[test_idx]
                train_labels, test_labels = tmp_label[train_idx], tmp_label[test_idx]

                # append train_files array to train_splits[split_id]
                train_splits[split_id].append(train_files)
                train_label_splits[split_id].append(train_labels)
                test_splits[split_id].append(test_files)
                test_label_splits[split_id].append(test_labels)

        # Concat the "emotion" array within the train_splits[0], train_splits[1] etc.
        for idx in range(len(train_splits)):
            train_splits[idx] = np.concatenate(train_splits[idx])
            train_label_splits[idx] = np.concatenate(train_label_splits[idx])
            test_splits[idx] = np.concatenate(test_splits[idx])
            test_label_splits[idx] = np.concatenate(test_label_splits[idx])

        # shuffle train and test data
        for idx in range(len(train_splits)):
            train_splits[idx], train_label_splits[idx] = shuffle(train_splits[idx], train_label_splits[idx])
            test_splits[idx], test_label_splits[idx] = shuffle(test_splits[idx], test_label_splits[idx])

        # save train/test to train_ids_{i}.csv/test_ids_{i}.csv
        train_filename = "train_ids_"
        test_filename = "test_ids_"
        for idx, split in enumerate(train_splits):
            train_dict = {'imgpath': split,
                          'emotion': train_label_splits[idx]}
            train_df = pd.DataFrame(train_dict, columns=['imgpath', 'emotion'])
            print("TESTEST", test_splits[idx])
            test_dict = {'imgpath': test_splits[idx],
                         'emotion': test_label_splits[idx]}
            test_df = pd.DataFrame(test_dict, columns=['imgpath', 'emotion'])
            # save to csv file now
            trf = self.args.affectnet + train_filename + "{0}.csv".format(idx)
            tef = self.args.affectnet + test_filename + "{0}.csv".format(idx)
            train_df.to_csv(trf, index=False)
            test_df.to_csv(tef, index=False)
            self.train_dfs.append(train_df)
            self.test_dfs.append(test_df)








if __name__ == '__main__':
    args = Setup().parse()
    args.affectnet_manual = "../../datasets/affectnet/training.csv"
    args.affectnet = "../../datasets/affectnet/"
    afnet = AffectNetUtils(args)

"""
if __name__ == '__main__':
    args = Setup().parse()
    args.fer_images = "../../datasets/fer/fer2013.csv"
    fer = FERUtils(args)
    data = fer.dataset.values
    print(fer.dataset['Usage'].value_counts())
    labels = data[:, 0]
    pix = data[:, 1]
    print("test: ", pix)

    # create array of all images with shape (n_images, n_pixels)
    images = np.zeros((pix.shape[0], 48*48))

    print(images.shape)

    for idx in range(images.shape[0]):
        p = pix[idx].split(' ')
        for idy in range(images.shape[1]):
            images[idx, idy] = int(p[idy])

    print(images)
    print(labels)

    for idx in range(4):
        plt.figure(idx)
        plt.imshow(images[idx].reshape((48, 48)), interpolation='none', cmap='gray')
        plt.show()

"""
