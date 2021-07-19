import numpy as np
import csv
import pandas as pd
from args2 import Setup
from sklearn.model_selection import KFold
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


class FERUtils:
    """
    # anger, disgust, fear, happy, sad, surprise, neutral
    # emotions = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}
    """
    def __init__(self, args, remove_label = None):
        self.args = args
        self.dataset = None
        self.train = None
        self.test = None
        self.val = None
        self.__load_csv__(args, remove_label)

    def __load_csv__(self, args, remove_label = None):
        self.dataset = pd.read_csv(args.fer_images)
        if isinstance(remove_label, int):
            self.dataset = self.dataset[self.dataset["emotion"] != remove_label]
        self.train = self.dataset.loc[self.dataset['Usage'] == "Training"]
        self.test = self.dataset.loc[self.dataset['Usage'] == "PublicTest"]
        self.val = self.dataset.loc[self.dataset['Usage'] == "PrivateTest"]

    def save_to_csv(self):
        # first we try without cross validation
        train_filename = "train_ids_0.csv"
        test_filename = "test_ids_0.csv"
        val_filename = "val_ids_0.csv"

        self.train.to_csv(self.args.fer + train_filename, index=False)
        self.test.to_csv(self.args.fer + test_filename, index=False)
        self.val.to_csv(self.args.fer + val_filename, index=False)

    def show_sample(self):
        pass


class AffectNetUtils():
    """
    training.csv is the manual annotated data
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


def run_affectnet_csv():
    args = Setup().parse()
    args.affectnet_manual = "../../datasets/affectnet/training.csv"
    args.affectnet = "../../datasets/affectnet/"
    afnet = AffectNetUtils(args)


class RafDBUtils:
    def __init__(self, args, remove_label = None):
        self.args = args
        self.__load_labelfile__(path_to_file=self.args.rafdb_labelfile)
        self.save_to_csv()

    def __load_labelfile__(self, path_to_file):
        self.file_names, self.labels = [], []
        self.train_files, self.train_labels = [], []
        self.test_files, self.test_labels = [], []
        for line in open(path_to_file):
            row = line.split("\n")
            name, label = row[0].split(" ")
            # sort for train
            if name.find("train") == -1:  # string train not found
                self.test_files.append(name)
                self.test_labels.append(label)
            else:
                self.train_files.append(name)
                self.train_labels.append(label)
            self.file_names.append(name)
            self.labels.append(label)


    def save_to_csv(self):
        train_filename = "train_ids_0.csv"
        test_filename = "test_ids_0.csv"
        val_filename = "valid_ids_0.csv"

        # split the test set to test/val 50/50
        X_test, X_val, y_test, y_val = train_test_split(self.test_files, self.test_labels, test_size=0.5, random_state=42)

        # create dataframes
        train_df = pd.DataFrame(data=list(zip(self.train_files, self.train_labels)), columns=['file_name', 'label'])
        test_df = pd.DataFrame(data=list(zip(X_test, y_test)), columns=['file_name', 'label'])
        val_df = pd.DataFrame(data=list(zip(X_val, y_val)), columns=['file_name', 'label'])

        train_df.to_csv(self.args.rafdb + train_filename, index=False)
        test_df.to_csv(self.args.rafdb + test_filename, index=False)
        val_df.to_csv(self.args.rafdb + val_filename, index=False)







"""
if __name__ == '__main__':
    args = Setup().parse()
"""


if __name__ == '__main__':
    args = Setup().parse()
    # args.fer_images = "../../datasets/fer/fer2013.csv"
    # args.fer = "../../datasets/fer/"
    # create fer dataset but remove label neutral
    # fer = FERUtils(args)
    # fer.save_to_csv()

    # data = fer.dataset.values
    # print(fer.dataset['Usage'].value_counts())
    # labels = data[:, 0]
    # pix = data[:, 1]
    # print("test: ", pix)
    args.rafdb_labelfile = "D:/datasets/RAFDB/basic/basic/EmoLabel/list_patition_label.txt"
    args.rafdb = "D:/projects/pyprojects/fer_fmpn_pytorch/datasets/rafdb/"
    raf = RafDBUtils(args=args)




    # Training
    # PrivateTest
    # PublicTest

"""
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