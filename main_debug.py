from lib.dataloader.datasets import CKP, Normalization, ToTensor, get_ckp
from lib.utils import imshow_tensor
from args2 import Setup
import numpy as np
import cv2
from sklearn.model_selection import KFold, train_test_split
import pandas
from lib.eval.eval_utils import make_ds_distribution_plot


def print_dataset(data):
    shapes = []
    for im in data:
        cv2.imshow("image", im["image"])
        cv2.waitKey()
        cv2.imshow("image", im["image_gray"])
        # print("img path", im['img_path'])
        cv2.waitKey()
        nrmimg = ToTensor()(im)
        nrmimg = Normalization()(nrmimg)
        cv2.imshow("image", nrmimg["image"].detach().cpu().numpy()[0])
        cv2.waitKey()
        if np.shape(im["image_gray"]) not in shapes:
            shapes.append(np.shape(im["image_gray"]))
    print(shapes)


def subjstr_to_csv(train_subjects, test_subjects, subjects_images, path, idx, valid_subjects=None):
    """
    :param train_subjects: List[str]
    :param test_subjects: List[str]
    :param subjects_images: Dict
    :return:
    """
    train_images, test_images, valid_images = [], [], []

    filename_train = "train_ids_{}.csv".format(idx)
    filename_test = "test_ids_{}.csv".format(idx)
    filename_valid = "valid_ids_{}.csv".format(idx)

    for subject in train_subjects:
        for key in subjects_images[subject]:
            for filename in subjects_images[subject][key]:
                train_images.append(filename)
    for subject in test_subjects:
        for key in subjects_images[subject]:
            for filename in subjects_images[subject][key]:
                test_images.append(filename)
    if valid_subjects is not None:
        for subject in valid_subjects:
            for key in subjects_images[subject]:
                for filename in subjects_images[subject][key]:
                    valid_images.append(filename)

    # save to csv files
    train_df = pandas.DataFrame(train_images)
    test_df = pandas.DataFrame(test_images)
    train_df.to_csv(path + filename_train, index=False, header=False)
    test_df.to_csv(path + filename_test, index=False, header=False)

    if valid_subjects is not None:
        valid_df = pandas.DataFrame(valid_images)
        valid_df.to_csv(path+filename_valid, index=False, header=False)



def create_splits():
    args = Setup().parse()

    # gets the subject id and filename
    subject_nr = lambda x: x[0:4]
    image_filename = lambda x: x[9:30]

    ckp_train = CKP(train=True, args=args, transform=False)
    ckp_test = CKP(train=False, args=args, transform=False)

    subjects = {}  # store subject dicts with nr of images per class
    subjects_images = {}  # store subject dicts with imagenames

    # add subject dicts for counting
    for im in ckp_train:
        nr = subject_nr(im["img_path"])
        label = im["label"]
        subjects[nr] = {
            '0': 0,
            '1': 0,
            '2': 0,
            '3': 0,
            '4': 0,
            '5': 0,
            '6': 0
        }
        subjects_images[nr] = {
            '0': [],
            '1': [],
            '2': [],
            '3': [],
            '4': [],
            '5': [],
            '6': []
        }

    # counter nr of images per subject and per class
    for im in ckp_train:
        nr = subject_nr(im["img_path"])
        filename = image_filename(im['img_path'])
        label = im["label"]
        subjects[nr][str(label)] += 1
        subjects_images[nr][str(label)].append(filename)

    print(subjects)
    print("Subjects nr: ", len(subjects))

    # for test set now
    for im in ckp_test:
        nr = subject_nr(im["img_path"])
        label = im["label"]
        subjects[nr] = {'0': 0,
                        '1': 0,
                        '2': 0,
                        '3': 0,
                        '4': 0,
                        '5': 0,
                        '6': 0}
        subjects_images[nr] = {'0': [],
                               '1': [],
                               '2': [],
                               '3': [],
                               '4': [],
                               '5': [],
                               '6': []}

    for im in ckp_test:
        nr = subject_nr(im["img_path"])
        filename = image_filename(im['img_path'])
        label = im["label"]
        subjects[nr][str(label)] += 1
        subjects_images[nr][str(label)].append(filename)

    print("Subjects nr.: ", len(subjects))
    print("Subjects imgs list nr.: ", len(subjects_images))
    print(subjects_images)

    nr_img = 0
    swan = []  # subjects with anger
    swco = []  # subjects with contempt
    swithoutco = []
    swdi = []  # subjects with disgust
    swfe = []  # subjects with fear
    swha = []  # subjects with happy
    swsa = []  # subjects with sadness
    swsu = []  # subjects with surprise

    # search for subjects providing at least contempt emotion
    for emo in [1]:  # 1 = contempt
        for s in subjects:
            if subjects[s][str(emo)] > 0:
                swco.append(s)
            else:
                swithoutco.append(s)
            nr_img += subjects[s][str(emo)]
    #
    # CREATE 10 fold datasets
    #
    splits_train, splits_test, splits_valid = [], [], []
    # while not classes_not_zero:
    kfold = KFold(n_splits=10, shuffle=True, random_state=8)  # 1,2,3,4

    for train, test in kfold.split(swco):
        # at first split all subjects who have contempt images
        splits_train.append([])
        splits_test.append([])
        for subid in train:
            splits_train[-1].append(swco[subid])
        for subid in test:
            splits_test[-1].append(swco[subid])

    for idx, fold in enumerate(splits_train):
        # split each train fold to train, validation fold 0.9/0.1 here
        # because at this time splits_train contains only subjects with
        # contempt emotion! We want each split to have all classes represented.
        train, valid = train_test_split(fold, test_size=0.1, shuffle=True)
        splits_train[idx] = train
        splits_valid.append(valid)

    for idx, (train, test) in enumerate(kfold.split(swithoutco)):
        # split subjects which do not have any contempt images
        # into 10 folds and append them to the corresponding idx
        # of splits_train, splits_test.
        # Before this will be done split the subjects without
        # contempt of the train split into train, val for each fold
        train, valid = train_test_split(train, test_size=0.1, shuffle=True)

        for subid in train:
            splits_train[idx].append(swithoutco[subid])
        for subid in test:
            splits_test[idx].append(swithoutco[subid])
        for subid in valid:
            splits_valid[idx].append(swithoutco[subid])

    split_indexes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    for idx, split in enumerate(split_indexes):
        # print(subjects_images)
        # print(splits_train[split])
        count_emotions_per_split(splits_test[split], subjects_images)

        subjstr_to_csv(train_subjects=splits_train[split],
                       test_subjects=splits_test[split],
                       valid_subjects=splits_valid[split],
                       subjects_images=subjects_images,
                       path="./datasets/ckp/tensplit_traintestval/",
                       idx=split)


def count_emotions_per_split(subjects_list, subject_images_dict):
    final_count = [0, 0, 0, 0, 0, 0, 0]
    for subject in subjects_list:
        count = count_images_of_subject(subject, subject_images_dict)
        for idx, num in enumerate(count):
            final_count[idx] += num
    print(final_count)


def count_images_of_subject(subject_key: str, subject_images_dict):
    """count the amount of images per class and return list containing counts"""
    count = [0, 0, 0, 0, 0, 0, 0]
    subject_dict = subject_images_dict[subject_key]
    for key in subject_dict:
        count[int(key)] = len(subject_dict[key])
    return count


def intersection_exists(split_a, split_b, split_c):
    return bool( set(split_a) & set(split_b) & set(split_c) )


def classes_not_zero():
    pass


def plot_dist(args):
    for i in range(10):
         args.trainsplit = "train_ids_{}.csv".format(i)
         args.testsplit = "test_ids_{}.csv".format(i)
         args.validsplit = "valid_ids_{}.csv".format(i)

         train_dl, test_dl, valid_dl = get_ckp(args=args, valid=True)

         make_ds_distribution_plot(train_dl=train_dl,
                                   test_dl=test_dl,
                                   valid_dl=valid_dl,
                                   save_to="./datasets/ckp/tensplit/split_{}_".format(i),
                                   n_classes=7)


if __name__ == "__main__":
    args = Setup().parse()

    # ckp_train = CKP(train=True, args=args, transform=False)
    # ckp_test = CKP(train=False, args=args, transform=False)
    # ckp_valid = CKP(valid=True, train=False, args=args)

    # create_splits()


    plot_dist(args)





"""
    nrmimg = ToTensor()(ckp[0])
    nrmimg = Normalization()(nrmimg)

    cv2.imshow("image", ckp[0]["image"])
    cv2.waitKey()
    # cv2.imshow("image", nrmimg["image"])
    cv2.imshow("image", nrmimg["image"].detach().cpu().numpy()[0])
    cv2.waitKey()

"""