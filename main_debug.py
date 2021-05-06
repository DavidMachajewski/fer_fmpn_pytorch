from lib.dataloader.datasets import CKP, Normalization, ToTensor, get_ckp
from lib.utils import imshow_tensor
from args2 import Setup
import numpy as np
import cv2
from sklearn.model_selection import KFold
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


def subjstr_to_csv(train_subjects, test_subjects, subjects_images, path, idx):
    """
    :param train_subjects: List[str]
    :param test_subjects: List[str]
    :param subjects_images: Dict
    :return:
    """
    filename_train = "train_ids_{}.csv".format(idx)
    filename_test = "test_ids_{}.csv".format(idx)
    train_images = []
    test_images = []
    for subject in train_subjects:
        for key in subjects_images[subject]:
            print("key: ", type(key))
            for filename in subjects_images[subject][key]:
                train_images.append(filename)
    for subject in test_subjects:
        for key in subjects_images[subject]:
            for filename in subjects_images[subject][key]:
                test_images.append(filename)
    print(train_images)
    print(len(train_images))
    print(test_images)
    print(len(test_images))

    # save to csv files
    train_df = pandas.DataFrame(train_images)
    test_df = pandas.DataFrame(test_images)
    train_df.to_csv(path+filename_train, index=False, header=False)
    test_df.to_csv(path+filename_test, index=False, header=False)


def create_splits():
    args = Setup().parse()

    subject_nr = lambda x: x[0:4]  # gets the subject id
    image_filename = lambda x: x[9:30]

    ckp_train = CKP(train=True, args=args, transform=False)
    ckp_test = CKP(train=False, args=args, transform=False)

    subjects = {}  # store subject dicts with nr of images per class
    subjects_images = {}  # store subject dicts with imagenames

    # add subject dicts for counting
    for im in ckp_train:
        nr = subject_nr(im["path"])
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

    # counter nr of images per subject and per class
    for im in ckp_train:
        nr = subject_nr(im["path"])
        filename = image_filename(im['path'])
        label = im["label"]
        subjects[nr][str(label)] += 1
        subjects_images[nr][str(label)].append(filename)

    print(subjects)
    print("Subjects nr: ", len(subjects))

    # for test set now
    for im in ckp_test:
        nr = subject_nr(im["path"])
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
        nr = subject_nr(im["path"])
        filename = image_filename(im['path'])
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

    for emo in [1]:  # contempt
        for s in subjects:
            if subjects[s][str(emo)] > 0:
                swco.append(s)
            else:
                swithoutco.append(s)
            nr_img += subjects[s][str(emo)]

    print("Nr. of contempt images: ", nr_img)
    print("Nr. of subjects with contempt images: ", len(swco))
    print("Subjects with contempt images: ", swco)
    print("Nr. of subjects without contempt images: ", len(swithoutco))
    print("Subjects without contempt images: ", swithoutco)

    #
    # CREATE 10 fold datasets
    #
    splits_train = []
    splits_test = []
    #while not classes_not_zero:
    kfold = KFold(n_splits=10, shuffle=True, random_state=3)  # 1,2,3,4
    print(swco)
    for train, test in kfold.split(swco):
        # at first split all subjects who have contempt images
        print("Train contempt: ", train)
        print("Test contempt : ", test)
        splits_train.append([])
        splits_test.append([])
        for subid in train:
            splits_train[-1].append(swco[subid])
        for subid in test:
            splits_test[-1].append(swco[subid])

    for idx, (train, test) in enumerate(kfold.split(swithoutco)):
        # split subjects who do not have any contempt images
        print("Train: ", train)
        print("Test: ", test)
        for subid in train:
            splits_train[idx].append(swithoutco[subid])
        for subid in test:
            splits_test[idx].append(swithoutco[subid])

    split_indexes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    for split in split_indexes:
        subjstr_to_csv(splits_train[split], splits_test[split], subjects_images, "./datasets/ckp/tensplit_new/", split)


def classes_not_zero():
    pass


if __name__ == "__main__":
    args = Setup().parse()

    # ckp_train = CKP(train=True, args=args, transform=False)
    # ckp_test = CKP(train=False, args=args, transform=False)

    # create_splits()

    for i in range(10):
         args.trainsplit = "train_ids_{}.csv".format(i)
         args.testsplit = "test_ids_{}.csv".format(i)

         train_dl, test_dl = get_ckp(args)

         make_ds_distribution_plot(train_dl=train_dl,
                                   test_dl=test_dl,
                                   save_to="./datasets/ckp/tensplit/split_{}_".format(i),
                                   n_classes=7)




"""
    nrmimg = ToTensor()(ckp[0])
    nrmimg = Normalization()(nrmimg)

    cv2.imshow("image", ckp[0]["image"])
    cv2.waitKey()
    # cv2.imshow("image", nrmimg["image"])
    cv2.imshow("image", nrmimg["image"].detach().cpu().numpy()[0])
    cv2.waitKey()

"""
