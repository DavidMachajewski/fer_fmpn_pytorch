import os
import pandas as pd
import torch as to
from torch.utils.data import Dataset, DataLoader
import pickle
import cv2 as cv
import numpy as np
import torchvision as tv
import torchvision.transforms as transforms


class DatasetBase(Dataset):
    def __init__(self, args, train: bool, valid: bool):
        """
        :param args:
        :param train: 1 -> train, 0 -> test
        :param valid:
        """
        self.args = args
        self.train = train
        self.valid = valid
        super(DatasetBase, self).__init__()
        self.data_root = self.args.data_root
        self.masks_path = self.args.masks

        if train:
            self.splitname = args.trainsplit
        elif not train and not valid:
            self.splitname = args.testsplit
        elif valid:
            self.splitname = args.validsplit

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass

    def __load_image__(self, path):
        # print(path)
        img = cv.imread(path)
        if self.train:
            # load bigger size image to apply random cropping while training ckp on fmpn/fmg (small dataset)
            img = cv.resize(img, (self.args.load_size, self.args.load_size))
        else:
            img = cv.resize(img, (self.args.final_size, self.args.final_size))
        if img.shape[2] == 1:
            # to get 3 channel image
            img = cv.cvtColor(img, cv.CV_GRAY2RGB)
            # dividing by 255.
        img = cv.normalize(img, None, alpha=0, beta=1, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)
        # img = 1./255. * img
        return img

    def __load_mask__(self, label):
        """Loading the facial mask corresponding to the label"""
        if self.train or self.valid:
            idx = 10  # index of fold nr. within the filename of splits
        else:
            idx = 9
        get_id = lambda id_name: self.splitname[idx]
        path_to_masks = self.args.masks

        mask_sub_folder = "train_{}".format(get_id(self.splitname))
        mask_file_name = "mask_{}.png".format(label + 1)  # + 1 because of names id are label+1

        mask_path = os.path.join(path_to_masks, mask_sub_folder, mask_file_name)

        img = cv.imread(mask_path)

        img = cv.resize(img, (self.args.final_size, self.args.final_size))
        # grayscale
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        norm_img = cv.normalize(img, None, alpha=0, beta=1, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)

        norm_img = np.expand_dims(norm_img, axis=2)
        return norm_img


class RafDB(DatasetBase):
    """
    Classes:
    1: Surprise
    2: Fear
    3: Disgust
    4: Happiness
    5: Sadness
    6: Anger
    7: Neutral

    convert them to
    0 surprise, 1 fear, 2 disgust, 3 happiness, 4 sadness, 5 anger, 6 neutral

    ckp labels for loading masks
    0: anger
    1: contempt
    2: disgust
    3: fear
    4: happiness
    5: sadness
    6: surprise

    """

    def __init__(self, args, train: bool, transform=None, valid=False, ckp_label_type=False, remove_class=None):
        """
        :param args:
        :param train:
        :param transform:
        :param valid:
        :param ckp_label_type: used for loading the mask if training fmpn network
        :param remove_class: original class label of emotion to remove from data
        """
        self.train = train
        self.transform = transform
        self.remove_class = remove_class
        self.ckp_label_type = ckp_label_type
        super(RafDB, self).__init__(args, train=train, valid=valid)
        self.__load_file__()
        self.__remove_emotion__()

    def __remove_emotion__(self):
        """remove a given emotion from the dataset"""
        # print("remove class nr. {}".format(self.remove_class))
        if isinstance(self.remove_class, int):
            print("Removing class nr. {}".format(self.remove_class))
            self.data = self.data[self.data["label"] != self.remove_class]

    def convert_label(self, label):
        # modify label by subtracting 1
        return label - 1

    def convert_label_to_masklabel(self, label):
        """convert modified label to ckp mask labels for loading the specific mask jpg."""
        # only use this if you want to load masks for training fmpn network
        ckplabels = [6, 3, 2, 4, 5, 0, -1]
        return ckplabels[label]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        label = self.data.iloc[[idx]]["label"].values
        file_name = self.data.iloc[[idx]]["file_name"].values[0]

        if self.train:
            file_name = file_name[0:11] + "_aligned" + file_name[11:15]
        elif not self.train and not self.valid:
            file_name = file_name[0:9] + "_aligned" + file_name[9:14]
        elif self.valid and not self.train:
            file_name = file_name[0:9] + "_aligned" + file_name[9:14]

        img_path = self.args.rafdb_imgs + file_name

        img = self.__load_image__(img_path)  # load image of size (320,320,3) = (load_size, load_size, n_channels)

        # resize image to final_size
        # img = cv.resize(img, dsize=(self.args.final_size, self.args.final_size), interpolation=cv.INTER_CUBIC)

        img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        # no resizing needed here because img_gray is a new var of img, but grayscaled
        # img_gray = cv.resize(img_gray, dsize=(self.args.load_size, self.args.load_size), interpolation=cv.INTER_CUBIC)
        img_gray = np.expand_dims(img_gray, axis=-1)  # (W;H;1)

        # convert label to ckp label
        label = self.convert_label(label)

        if self.ckp_label_type:  # use this if you want to train the fmpn
            mask_label = self.convert_label_to_masklabel(label[0])

            mask = self.__load_mask__(mask_label)
            sample = {'image': img,
                      'image_gray': img_gray,
                      'label': label[0],
                      'mask': mask,
                      'img_path': img_path}
        else:
            sample = {'image': img,
                      'image_gray': img_gray,
                      'label': label[0]}

        if self.transform:
            sample = self.transform(sample)
        return sample

    def __load_file__(self):
        """load the .csv file"""
        path = self.args.rafdb + self.splitname
        print("Loading {0} from {1}".format(self.splitname, path))
        self.data = pd.read_csv(path)


class CKP(DatasetBase):
    """
    self.classes = {
            "1": "anger",
            "2": "contempt",
            "3": "disgust",
            "4": "fear",
            "5": "happiness",
            "6": "sadness",
            "7": "surprise"
        }

    the dataloader reduces by 1 so the labels are
    0: anger
    1: contempt
    2: disgust
    3: fear
    4: happiness
    5: sadness
    6: surprise
    """

    def __init__(self, args, train, transform=None, valid=False):
        super(CKP, self).__init__(args, train=train, valid=valid)
        self.args = args
        self.transform = transform
        self.images_path = args.ckp_images
        self.path_csv = os.path.join(args.ckp_csvsplits, self.splitname)

        self.img_names = pd.read_csv(self.path_csv, index_col=False, header=None).values.tolist()
        # print("image names: ", self.img_names)
        self.img_names = [img_name[0] for img_name in self.img_names]
        self.img_paths, self.img_labels = self.__get_paths_and_labels__()

    def __get_paths_and_labels__(self):
        get_tag = lambda name: name[0:8]
        ds_tags = list(map(get_tag, self.img_names))

        ds_labels = pickle.load(open(self.args.labels, "rb"))

        get_tags = lambda name: (name[0:4], name[5:8])
        img_paths, labels = [], []
        for i in range(self.__len__()):
            imgname = self.img_names[i]
            tags = get_tags(imgname)
            imgtag = ds_tags[i]
            image_path = os.path.join(tags[0], tags[1], imgname)
            label = int(ds_labels[imgtag]) - 1  # -1 because CK+ has labels 1-7 -> convert to 0-6
            img_paths.append(image_path)
            labels.append(label)
        return img_paths, labels

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        #print(idx)
        img_path = self.img_paths[idx]
        #print(img_path)
        label = self.img_labels[idx]
        #print(label)
        # load image
        img = self.__load_image__(self.images_path + img_path)
        img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        img_gray = np.expand_dims(img_gray, axis=-1)  # (W;H;1)
        mask = self.__load_mask__(label)

        sample = {'image': img,
                  'image_gray': img_gray,
                  'label': label,
                  'mask': mask,
                  'img_path': img_path}
        # for debugging you can add image path to the dict

        if self.transform:
            sample = self.transform(sample)
        return sample


class FER2013(DatasetBase):
    """
    ferlabels

    0: "anger",
    1: "disgust",
    2: "fear",
    3: "happiness",
    4: "sadness",
    5: "surprise",
    6: "neutral"

    ckplabels
    0: anger 1: contempt  2: disgust 3: fear  4: happiness 5: sadness 6: surprise

    fer to ckp (Für die masken notwendig)
    0 -> 0 : anger
    1 -> 2 : disgust
    2 -> 3 : fear
    3 -> 4 : happiness
    4 -> 5 : sadness
    5 -> 6 : surprise
    6 -> -1: not available ( rausnehmen aus dem Datenset )

    0 -> 0 : anger
    1 -> 1 : disgust
    2 -> 2 : fear
    3 -> 3 : happiness
    4 -> 4 : sadness
    5 -> 5 : surprise
    """

    def __init__(self, args, train: bool, transform=None, valid=False, ckp_label_type=False, remove_class=None):
        """
        :param args:
        :param train:
        :param transform:
        :param valid:
        :param ckp_label_type:
        :param remove_class: remove class 6 if you do not need the neutral label (e.g. for training fmpn)
        """
        self.train = train
        self.transform = transform
        self.remove_class = remove_class
        super(FER2013, self).__init__(args, train=train, valid=valid)
        self.data = None
        self.ckp_label_type = ckp_label_type
        self.__load_file__()
        self.__remove_emotion__()

    def __remove_emotion__(self):
        """remove a given emotion from the dataset"""
        # print("remove class nr. {}".format(self.remove_class))
        if isinstance(self.remove_class, int):
            print("Removing class nr. {}".format(self.remove_class))
            self.data = self.data[self.data["emotion"] != self.remove_class]

    def __load_file__(self):
        path = self.args.fer + self.splitname
        # print("Loading {0} from {1}".format(self.splitname, path))
        self.data = pd.read_csv(path)

    def get_img_as_2d_array(self, pixels):
        # convert pixels which are saved as string
        # within the dataframe to image of size 48x48
        image = np.zeros((1, 48 * 48))
        for idx in range(image.shape[0]):
            p = pixels[idx].split(' ')
            for idy in range(image.shape[1]):
                image[idx, idy] = int(p[idy])
        return image.reshape((48, 48))

    def convert_label(self, fer_label):
        new_labels = [0, 1, 2, 3, 4, 5, 6]
        return new_labels[fer_label]

    def convert_label_to_masklabel(self, fer_label):
        """convert fer label to ck+ label type
        to load the correct mask or make it more compareable at least"""
        # fer in ckp form anger, disgust, fear, happiness, sadness, surprise
        ckplabels = [0, 2, 3, 4, 5, 6]
        return ckplabels[fer_label]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        label = self.data.iloc[[idx]]["emotion"].values

        pixels = self.data.iloc[[idx]]["pixels"].values

        img_gray = self.get_img_as_2d_array(pixels)
        # img = np.expand_dims(img, axis=-1)
        # img = img * 1./255.
        img_gray = cv.normalize(img_gray, None, alpha=0, beta=1, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)
        img = cv.cvtColor(img_gray, cv.COLOR_GRAY2RGB)

        if self.train:
            img_gray = cv.resize(img_gray, dsize=(self.args.load_size, self.args.load_size),
                                 interpolation=cv.INTER_CUBIC)
            img = cv.resize(img, dsize=(self.args.load_size, self.args.load_size), interpolation=cv.INTER_CUBIC)
        else:
            img_gray = cv.resize(img_gray, dsize=(self.args.final_size, self.args.final_size),
                                 interpolation=cv.INTER_CUBIC)
            img = cv.resize(img, dsize=(self.args.final_size, self.args.final_size), interpolation=cv.INTER_CUBIC)

        img_gray = np.expand_dims(img_gray, axis=-1)  # (W;H;1)

        if self.ckp_label_type:  # use this if you want to train the fmpn
            mask = self.__load_mask__(self.convert_label_to_masklabel(label[0]))

        # sample = {'image': img,
        #           'image_gray': img_gray,
        #           'label': label}

        if self.ckp_label_type:  # label type for training fmpn
            sample = {'image': img,  # 3 channel image
                      'image_gray': img_gray,  # 1 channel image
                      'label': self.convert_label(label[0]),
                      'label_to_load_mask': self.convert_label_to_masklabel(label[0]),
                      'mask': mask}  # get ckp explicit emotion to load correct mask
        else:  # original label type without neutral
            sample = {'image': img,
                      'image_gray': img_gray,
                      'label': self.convert_label(label[0])}

        if self.transform:
            sample = self.transform(sample)
        return sample


class AffectNet(DatasetBase):
    """
    0: neutral
    1: happiness    0
    2: sadness      1
    3: surprise     2
    4: fear         3
    5: disgust      4
    6: anger        5
    7: contempt     6

    8: None
    9: Uncertain
    10: No-Face
    """

    def __init__(self, args, train: bool, transform=None, valid=False, ckp_label_type=False, remove_class=None):
        """To remove the neutral emotion, use remove_class=0"""
        super(AffectNet, self).__init__(args, train, valid)
        self.train = train
        self.transform = transform
        self.remove_class = remove_class
        self.ckp_label_type = ckp_label_type
        self.__load_file__()
        self.__remove_emotion__()
        #
        # paths to image folders into args
        #

    def __load_file__(self):
        path = self.args.affectnet + self.splitname
        print("Loading {0} from {1}".format(self.splitname, path))
        self.data = pd.read_csv(path)

    def __remove_emotion__(self):
        """remove a given emotion from the dataset"""
        # print("remove class nr. {}".format(self.remove_class))
        if isinstance(self.remove_class, int):
            print("Removing class nr. {}".format(self.remove_class))
            self.data = self.data[self.data["label"] != self.remove_class]

    def convert_label_to_masklabel(self, label):
        """convert modified label to ckp mask labels for loading the specific mask jpg."""
        # only use this if you want to load masks for training fmpn network
        """
        ckp labels for loading masks
        0: anger
        1: contempt
        2: disgust
        3: fear
        4: happiness
        5: sadness
        6: surprise
        """
        ckplabels = [4, 5, 6, 3, 2, 0, 1]
        return ckplabels[label]

    def convert_label(self, label):
        # because neutral is 0 and other labels start at 1
        # so if we remove label neutral there will be a bug
        # in n_classes and label idxs not matching
        # USE THIS FOR TRAINING THE FMPN
        return label - 1

    def __len__(self):
        return len(self.data)  # check if this works for dataframes!

    def __getitem__(self, idx):
        label = self.data.iloc[[idx]]["label"].values
        # the file_name contains the subpath like /Manually_Anotated_compressed/600/cfa0... .jpg
        file_name = self.data.iloc[[idx]]["file_name"].values[0]

        if self.train:
            # print("Filename prefix: ", file_name[0:7])
            if file_name[0:8] == "Manually":
                img_path = self.args.affectnet_img_parentfolder_man + file_name
            else:  # Automatically
                img_path = self.args.affectnet_img_parentfolder_aut + file_name
        elif not self.train and not self.valid:
            img_path = self.args.affectnet_img_parentfolder_man + file_name
        elif self.valid and not self.train:
            img_path = self.args.affectnet_img_parentfolder_man + file_name

        # img_path = self.args.rafdb_imgs + file_name

        img = self.__load_image__(img_path)  # load image of size (320,320,3) = (load_size, load_size, n_channels)
        img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        img_gray = np.expand_dims(img_gray, axis=-1)  # (W;H;1)

        # convert label to ckp label
        if self.ckp_label_type:  # ckp_label_type implies that neutral class is removed
            label = self.convert_label(label)
            mask_label = self.convert_label_to_masklabel(label[0])
            mask = self.__load_mask__(mask_label)
            sample = {'image': img,
                      'image_gray': img_gray,
                      'label': label[0],
                      'mask': mask,
                      'img_path': img_path}
        else:
            sample = {'image': img,
                      'image_gray': img_gray,
                      'label': label[0]}

        if self.transform:
            sample = self.transform(sample)
        return sample


class AffectNetSubset(DatasetBase):
    #
    # Check emotion categories for affect net
    # and provide to this class
    #
    def __init__(self, args, train: bool, transform):
        self.args = args
        self.transform = transform
        super(AffectNetSubset, self).__init__(args=args, train=train)
        self.path_csv = os.path.join(args.affectnet, self.splitname)
        self.data = pd.read_csv(self.path_csv, index_col=False, header=0)
        if train:
            self.splitname = args.trainsplit
        else:
            self.splitname = args.testsplit

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        imgsubpath = self.data.iloc[idx]['imgpath']
        label = self.data.iloc[idx]['emotion']

        # create/get image path
        path = self.args.affectnet_images + imgsubpath

        # load image
        image = self.__load_image__(path)

        # convert image to gray
        # image_gray = ...

        # load mask
        # mask = ...

        sample = {'image': image,
                  # 'image_gray': img_gray,
                  'label': label,
                  # 'mask': mask
                  }
        if self.transform:
            sample = self.transform(sample)
        return sample


def get_ckp(args, batch_size=8, shuffle=True, num_workers=2, drop_last=False, valid: bool = None):
    """Initialize ckp dataset and return train, test
    torch.utils.data.dataloader.DataLoader, already batched and shuffled."""
    # for loading the dataset set conditions for processing
    #
    #   RandomCrop: True, False
    #   RandomFlip: True, False
    #   Normalization: True, False
    if valid is None:
        valid = False
    transforms = tv.transforms.Compose([RandomCrop(args), ToTensor(), RandomFlip(), Normalization(args)])
    transforms_test = tv.transforms.Compose([ToTensor(), Normalization(args)])
    train_ds = CKP(train=True, args=args, transform=transforms)
    test_ds = CKP(train=False, args=args, transform=transforms_test)

    train_loader = DataLoader(dataset=train_ds,
                              batch_size=batch_size,
                              shuffle=shuffle,
                              num_workers=num_workers,
                              drop_last=drop_last)

    test_loader = DataLoader(dataset=test_ds,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             num_workers=num_workers)
    if valid:
        valid_ds = CKP(valid=True, train=False, args=args, transform=transforms_test)
        valid_loader = DataLoader(dataset=valid_ds,
                                  batch_size=batch_size,
                                  shuffle=shuffle,
                                  num_workers=num_workers)
        return train_loader, test_loader, valid_loader
    else:
        return train_loader, test_loader


def get_fer2013(args, batch_size=8, ckp_label_type=False, shuffle=True, num_workers=4, drop_last=False,
                augmentation=True, remove_class=None):
    if augmentation:
        print("Using data augmentation...")
        transforms = tv.transforms.Compose(
            [Fer2013RandomCrop(args), Fer2013ToTensor(), RandomFlip(), Fer2013Normalization(args)])
        transforms_test = tv.transforms.Compose([Fer2013ToTensor(), Fer2013Normalization(args)])
    else:
        transforms = tv.transforms.Compose([Fer2013ToTensor(), Fer2013Normalization(args)])
        transforms_test = tv.transforms.Compose([Fer2013ToTensor(), Fer2013Normalization(args)])

    train_ds = FER2013(train=True, args=args, transform=transforms, ckp_label_type=ckp_label_type,
                       remove_class=remove_class)
    test_ds = FER2013(train=False, args=args, transform=transforms_test, valid=False, ckp_label_type=ckp_label_type,
                      remove_class=remove_class)
    valid_ds = FER2013(train=False, args=args, transform=transforms_test, valid=True, ckp_label_type=ckp_label_type,
                       remove_class=remove_class)

    train_loader = DataLoader(dataset=train_ds,
                              batch_size=batch_size,
                              shuffle=shuffle,
                              num_workers=num_workers,
                              drop_last=drop_last)

    test_loader = DataLoader(dataset=test_ds,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             num_workers=num_workers,
                             drop_last=drop_last)

    valid_loader = DataLoader(dataset=valid_ds,
                              batch_size=batch_size,
                              shuffle=shuffle,
                              num_workers=num_workers,
                              drop_last=drop_last)

    return train_loader, test_loader, valid_loader


def get_rafdb(args, batch_size=8, ckp_label_type=False, shuffle=True, num_workers=4, drop_last=False,
              augmentation=False, remove_class=None):
    """

    :param args:
    :param batch_size:
    :param ckp_label_type:
    :param shuffle:
    :param num_workers:
    :param drop_last:
    :param augmentation:
    :param remove_class:
    :return:

            transforms = tv.transforms.Compose([Fer2013RandomCrop(args), Fer2013ToTensor(), RandomFlip(), Fer2013Normalization(args)])
        transforms_test = tv.transforms.Compose([Fer2013ToTensor(), Fer2013Normalization(args)])
    """
    if augmentation:
        print("Using data augmentation to process RAF-DB")
        # Fer2013RandomCrop should work here as well
        transforms = tv.transforms.Compose(
            [Fer2013RandomCrop(args), RafdbToTensor(), RandomFlip(), Fer2013Normalization(args)])
        transforms_test = tv.transforms.Compose([RafdbToTensor(), Fer2013Normalization(args)])
    else:
        transforms = tv.transforms.Compose([RafdbToTensor(), Fer2013Normalization(args)])
        transforms_test = tv.transforms.Compose([RafdbToTensor(), Fer2013Normalization(args)])

    train_ds = RafDB(train=True, args=args, transform=transforms, ckp_label_type=ckp_label_type,
                     remove_class=remove_class)
    test_ds = RafDB(train=False, args=args, transform=transforms_test, valid=False, ckp_label_type=ckp_label_type,
                    remove_class=remove_class)
    valid_ds = RafDB(train=False, args=args, transform=transforms_test, valid=True, ckp_label_type=ckp_label_type,
                     remove_class=remove_class)

    train_loader = DataLoader(dataset=train_ds,
                              batch_size=batch_size,
                              shuffle=shuffle,
                              num_workers=num_workers,
                              drop_last=drop_last)

    test_loader = DataLoader(dataset=test_ds,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             num_workers=num_workers,
                             drop_last=drop_last)

    valid_loader = DataLoader(dataset=valid_ds,
                              batch_size=batch_size,
                              shuffle=shuffle,
                              num_workers=num_workers,
                              drop_last=drop_last)

    return train_loader, test_loader, valid_loader


def get_affectnet(args, batch_size=8, ckp_label_type=False, shuffle=True, num_workers=4, drop_last=False, augmentation=False, remove_class=None, subset=False):
    if subset:
        transforms = tv.transforms.Compose([AffectNetToTensor()])
        transforms_test = tv.transforms.Compose([AffectNetToTensor()])
        # train_ds = AffectNetSubset(train=True, args=args, transform=transforms)
        # test_ds = AffectNetSubset(train=False, args=args, transform=transforms_test)
        #
        # valid_ds
        #
    else:
        transforms = tv.transforms.Compose([AffectNetToTensor()])
        transforms_test = tv.transforms.Compose([AffectNetToTensor()])
        train_ds = AffectNet(args=args, train=True, transform=transforms, ckp_label_type=ckp_label_type, remove_class=remove_class)
        test_ds = AffectNet(train=False, args=args, transform=transforms_test, valid=False, ckp_label_type=ckp_label_type, remove_class=remove_class)
        valid_ds = RafDB(train=False, args=args, transform=transforms_test, valid=True, ckp_label_type=ckp_label_type, remove_class=remove_class)


    train_loader = DataLoader(dataset=train_ds,
                              batch_size=batch_size,
                              shuffle=shuffle,
                              num_workers=num_workers,
                              drop_last=drop_last)

    test_loader = DataLoader(dataset=test_ds,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             num_workers=num_workers,
                             drop_last=drop_last)

    valid_loader = DataLoader(dataset=valid_ds,
                              batch_size=batch_size,
                              shuffle=shuffle,
                              num_workers=num_workers,
                              drop_last=drop_last)

    return train_loader, test_loader, valid_loader


# #################################################################
#
# callable classes, can be used like a layer to stack in a model
#
# #################################################################


class Normalization(object):
    def __init__(self, args):
        self.args = args

    def __call__(self, sample):
        image = sample['image']
        image = image.float()
        image_gray = sample['image_gray']
        mask = sample['mask']
        label = sample['label']
        path = sample['img_path']

        if self.args.norm_orig_img:
            image = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(image)

        return {'image': image,
                'image_gray': image_gray,
                'label': label,
                'mask': mask,
                'img_path': path}


class Fer2013Normalization(object):
    def __init__(self, args):
        self.args = args

    def __call__(self, sample):
        image = sample['image']
        label = sample['label']
        image_gray = sample['image_gray']

        image = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(image)

        if 'mask' in sample.keys():
            mask = sample['mask']
            return {'image': image,
                    'image_gray': image_gray,
                    'mask': mask,
                    'label': label}
        else:
            return {'image': image,
                    'label': label}


class Fer2013RandomCrop(object):
    def __init__(self, args):
        self.args = args

    def __call__(self, sample):
        image = sample['image']
        image_gray = sample['image_gray']
        label = sample['label']

        h, w = image.shape[:2]
        new_h, new_w = self.args.final_size, self.args.final_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h, left: left + new_w]
        image_gray = image_gray[top: top + new_h, left: left + new_w]

        if "mask" in sample.keys():
            mask = sample['mask']
            return {'image': image,
                    'image_gray': image_gray,
                    'label': label,
                    'mask': mask}
        else:
            return {'image': image,
                    'image_gray': image_gray,
                    'label': label}


class RandomCrop(object):
    """Crop image of given sample to final_size"""

    def __init__(self, args):
        self.args = args

    def __call__(self, sample):
        image = sample['image']
        image_gray = sample['image_gray']
        mask = sample['mask']
        label = sample['label']
        path = sample["img_path"]

        h, w = image.shape[:2]
        new_h, new_w = self.args.final_size, self.args.final_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h, left: left + new_w]
        image_gray = image_gray[top: top + new_h, left: left + new_w]
        return {'image': image,
                'image_gray': image_gray,
                'label': label,
                'mask': mask,
                'img_path': path}


class RandomFlip(object):
    """flip image and grayscaled image horizontally
    with a probability p"""

    def __call__(self, sample):
        image = sample['image']
        label = sample['label']
        image_gray = sample['image_gray']
        path = sample["img_path"]

        image = tv.transforms.RandomHorizontalFlip(p=0.5)(image)
        image_gray = tv.transforms.RandomHorizontalFlip(p=0.5)(image_gray)

        if 'mask' in sample.keys():
            mask = sample['mask']
            return {'image': image,
                    'image_gray': image_gray,
                    'label': label,
                    'mask': mask,
                    'img_path': path}
        else:
            return {'image': image,
                    'image_gray': image_gray,
                    'label': label}


class GrayScale(object):
    """Convert a 3 channel image tensor into a 1 channel image tensor"""

    def __call__(self, sample):
        image = sample['image']
        image = tv.transforms.Grayscale(num_output_channels=1)(image)
        path = sample["img_path"]

        label = sample['label']
        image_gray = sample['image_gray']
        mask = sample['mask']
        sample = {'image': image,
                  'image_gray': image_gray,
                  'label': label,
                  'mask': mask,
                  'img_path': path}
        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors with channel first format"""

    def __call__(self, sample):
        image = sample['image']
        image_gray = sample['image_gray']
        path = sample["img_path"]

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        image_gray = image_gray.transpose((2, 0, 1))

        label, mask = sample['label'], sample['mask']
        mask = mask.transpose((2, 0, 1))
        return {'image': to.from_numpy(image),
                'image_gray': to.from_numpy(image_gray),
                'label': to.tensor(label),
                'mask': to.from_numpy(mask),
                'img_path': path}


class AffectNetToTensor(object):
    def __call__(self, sample):
        image = sample['image']
        label = sample['label']
        image = image.transpose((2, 0, 1))
        return {'image': to.from_numpy(image),
                'label': to.tensor(label)}


class RafdbToTensor(object):
    def __call__(self, sample):
        image = sample['image']
        image_gray = sample['image_gray']
        label = sample['label']

        image = image.transpose((2, 0, 1))
        image_gray = image_gray.transpose((2, 0, 1))

        if 'mask' in sample.keys():
            mask = sample['mask']
            mask = mask.transpose((2, 0, 1))
            return {'image': to.from_numpy(image),
                    'image_gray': to.from_numpy(image_gray),
                    'mask': to.from_numpy(mask),
                    'label': to.tensor(label)}
        else:
            return {'image': to.from_numpy(image),
                    'image_gray': to.from_numpy(image_gray),
                    'label': to.tensor(label)}


class Fer2013ToTensor(object):
    def __call__(self, sample):
        image = sample['image']
        image_gray = sample['image_gray']
        label = sample['label']

        image = image.transpose((2, 0, 1))
        image_gray = image_gray.transpose((2, 0, 1))

        if "mask" in sample.keys():
            mask = sample['mask']
            mask = mask.transpose((2, 0, 1))
            return {'image': to.from_numpy(image),
                    'image_gray': to.from_numpy(image_gray),
                    'mask': to.from_numpy(mask),
                    'label': to.tensor(label)}

        return {'image': to.from_numpy(image),
                'image_gray': to.from_numpy(image_gray),
                'label': to.tensor(label)}


norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
