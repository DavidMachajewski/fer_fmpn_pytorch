import numpy as np
import csv
import os
import pandas as pd
from args2 import Setup
from sklearn.model_selection import KFold
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from lib.dataloader.datasets import RafDB


class FERUtils:
    """
    # anger, disgust, fear, happy, sad, surprise, neutral
    # emotions = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}
    """

    def __init__(self, args, remove_label=None):
        self.args = args
        self.dataset = None
        self.train = None
        self.test = None
        self.val = None
        self.__load_csv__(args, remove_label)

    def __load_csv__(self, args, remove_label=None):
        self.dataset = pd.read_csv(args.fer_images)
        if isinstance(remove_label, int):
            print("removing label nr: ", remove_label)
            self.dataset = self.dataset[self.dataset["emotion"] != remove_label]
        print("not removing label but saving whole dataset to 3 splits")
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
        self.dataset, self.dataset_aut, self.valid = self.__load_csv__()
        self.train_dfs, self.test_dfs, self.valid_dfs = [], [], []
        self.subfolder_manual = "Manually_Annotated_compressed/"
        self.subfolder_automatic = "Automatically_Annotated_compressed/"

    def __load_csv__(self):
        # load .csv files for training and validation (manual ann. val set) - split val into test/val
        dataset_manual = pd.read_csv(self.args.affectnet_manual)  # manually annotated csv file (training)
        dataset_automatically = pd.read_csv(self.args.affectnet_automatic_trainset)  # automatically annotated list
        validation_manual = pd.read_csv(
            self.args.affectnet_manual_valid)  # manually annotated validation csv file -> split it into test/validation
        return dataset_manual, dataset_automatically, validation_manual

    def __create_full_csv__(self):
        # merge dataset_manual and dataset_automatically and next split validation to test, valid
        # remove classes 8, 9, 10 from manually annotated dataset
        tmp_filenames, tmp_emotions = [], []
        self.dataset = self.dataset[self.dataset.iloc[:, 6] != 8]
        self.dataset = self.dataset[self.dataset.iloc[:, 6] != 9]
        self.dataset = self.dataset[self.dataset.iloc[:, 6] != 10]  # 8 9 10
        # remove classes 8, 9, 10 from automatically annotated dataset
        self.dataset_aut = self.dataset_aut[self.dataset_aut.iloc[:, 6] != 8]
        self.dataset_aut = self.dataset_aut[self.dataset_aut.iloc[:, 6] != 9]
        self.dataset_aut = self.dataset_aut[self.dataset_aut.iloc[:, 6] != 10]
        # remove classes 8, 9, 10 from manually validation set
        self.valid = self.valid[self.valid.iloc[:, 6] != 8]
        self.valid = self.valid[self.valid.iloc[:, 6] != 9]
        self.valid = self.valid[self.valid.iloc[:, 6] != 10]
        # actualize the paths to the parent subfolder to remove the ambiguity of subdirectory_filePath column
        for idx in range(0, len(self.dataset)):
            tmp_filenames.append(os.path.join(self.subfolder_manual, self.dataset.iat[idx, 0]))
            tmp_emotions.append(self.dataset.iat[idx, 6])
        self.dataset = pd.DataFrame(list(zip(tmp_filenames, tmp_emotions)), columns=['file_name', 'label'])

        tmp_filenames, tmp_emotions = [], []
        for idx in range(0, len(self.dataset_aut)):
            tmp_filenames.append(os.path.join(self.subfolder_automatic, self.dataset_aut.iat[idx, 0]))
            tmp_emotions.append(self.dataset_aut.iat[idx, 6])
        self.dataset_aut = pd.DataFrame(list(zip(tmp_filenames, tmp_emotions)), columns=['file_name', 'label'])

        # merge manual annotated dataset and automatically annotated dataset
        self.dataset = pd.concat([self.dataset, self.dataset_aut], axis=0)
        del self.dataset_aut

        tmp_filenames, tmp_emotions = [], []
        for idx in range(0, len(self.valid)):
            tmp_filenames.append(os.path.join(self.subfolder_manual, self.valid.iat[idx, 0]))
            tmp_emotions.append(self.valid.iat[idx, 6])
        self.valid = pd.DataFrame(list(zip(tmp_filenames, tmp_emotions)), columns=['file_name', 'label'])
        # split validation set into 50/50 test/validation sets
        msk = np.random.rand(len(self.valid)) <= 0.50
        self.test = self.valid[msk]
        self.valid = self.valid[~msk]
        # save to csv files
        self.dataset.to_csv(self.args.affectnet + "train_ids_0.csv", index=False)
        self.test.to_csv(self.args.affectnet + "test_ids_0.csv", index=False)
        self.valid.to_csv(self.args.affectnet + "valid_ids_0.csv", index=False)

    def __rd_sample__(self, emotions=[1, 2, 3, 4, 5, 6, 7], n_samples=None):
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
            tmp = tmp.sample(n=n_samples[str(emotions[idx])])
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
    # args.affectnet_manual = "../../datasets/affectnet/training.csv"
    # args.affectnet_automatic_trainset = ""
    # args.affectnet_manual_valid = ""
    args.affectnet = "../../datasets/affectnet/full/"
    afnet = AffectNetUtils(args)
    afnet.__create_full_csv__()


class RafDBUtils:
    def __init__(self, args, remove_label=None):
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
        X_test, X_val, y_test, y_val = train_test_split(self.test_files, self.test_labels, test_size=0.5,
                                                        random_state=42)

        # create dataframes
        train_df = pd.DataFrame(data=list(zip(self.train_files, self.train_labels)), columns=['file_name', 'label'])
        test_df = pd.DataFrame(data=list(zip(X_test, y_test)), columns=['file_name', 'label'])
        val_df = pd.DataFrame(data=list(zip(X_val, y_val)), columns=['file_name', 'label'])

        #
        # :TODO: REMOVE OBJECTIVE LABEL (NEUTRAL) FROM DATASETS
        #

        train_df.to_csv(self.args.rafdb + train_filename, index=False)
        test_df.to_csv(self.args.rafdb + test_filename, index=False)
        val_df.to_csv(self.args.rafdb + val_filename, index=False)


if __name__ == '__main__':
    # args = Setup().parse()
    # args.fer_images = "../../datasets/fer/fer2013.csv"
    # args.fer = "../../datasets/fer/"
    # create fer dataset but remove label neutral
    # fer = FERUtils(args)
    # fer.save_to_csv()
    run_affectnet_csv()

    # data = fer.dataset.values
    # print(fer.dataset['Usage'].value_counts())
    # labels = data[:, 0]
    # pix = data[:, 1]
    # # print("test: ", pix)
    # args.trainsplit = "train_ids_0.csv"
    # args.rafdb_labelfile = "D:/datasets/RAFDB/basic/basic/EmoLabel/list_patition_label.txt"
    # args.rafdb = "D:/projects/pyprojects/fer_fmpn_pytorch/datasets/rafdb/"
    # args.rafdb_imgs = "D:/projects/pyprojects/fer_fmpn_pytorch/datasets/rafdb/aligned/"
    # raf = RafDBUtils(args=args)

    # rafdb = RafDB(args=args, train=True, ckp_label_type=True)

    # print(rafdb[0])

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
