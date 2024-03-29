from args2 import Setup
import os
import cv2 as cv
import csv
import pandas as pd
from lib.dataloader.datasets import get_affectnet, AffectNet


def average_confusion_matrix(folder):
    """Create the average of n confusion matrices
    1. Find all paths to all cnfmat.txt files
    2. Load all cnf matrices one by one and add to a initial one
    3. Create the average confusion matrix like usually
    """
    pass


def clean_rafdb(path):
    """ This function removes all corrupted images from rafdb aligned folder
    and saves the corrupted image filenames to .txt file
    :param path: path to the aligned folder of the rafdb image
    """
    file_names = [file for file in os.listdir(path) if os.path.isfile(os.path.join(path, file))]
    corrupted_jpgs = []
    count_nones = 0

    # check if image is loadable
    for file in file_names:
        current_path = os.path.join(path, file)
        loaded_image = cv.imread(current_path)
        # add corrupted images to list and save
        if loaded_image is None:
            corrupted_jpgs.append(file)
            count_nones += 1
    # save corrupted images to file
    text_file = open(path + "corrupted_images.txt", "w")
    for file_name in corrupted_jpgs:
        text_file.write(file_name + "\n")
    # now delete all corrupted images from folder
    for file_name in corrupted_jpgs:
        corrupted_jpg_path = os.path.join(path, file_name)
        os.remove(corrupted_jpg_path)


def remove_corrputed_images_from_splits(path_to_corrputed, path_to_datasetcsv, save_to, file_name):
    """
    1. load corrupted_images.txt
    2. load all splits (train, test, validation)
    3. delete corrupted images from all splits
    :return:
    """
    files_to_delete = []
    with open(path_to_corrputed, 'r') as fd:
        reader = csv.reader(fd)
        for row in reader:
            files_to_delete.append(row[0])

    dataset_pd = pd.read_csv(path_to_datasetcsv)
    print(dataset_pd)

    for file in files_to_delete:
        if 'test' in file:
            file = file[0:9] + ".jpg"
        else:
            file = file[0:11] + ".jpg"
        print(file)
        dataset_pd = dataset_pd[dataset_pd['file_name'] != file]
    print(dataset_pd)

    dataset_pd.to_csv(os.path.join(save_to, file_name))


if __name__ == "__main__":
    args = Setup().parse()
    # clean_rafdb(args.rafdb_imgs)
    ptc = "D:\projects\pyprojects/fer_fmpn_pytorch\datasets/rafdb/aligned\corrupted_images.txt"
    train = "D:\projects\pyprojects/fer_fmpn_pytorch\datasets/rafdb/train_ids_0.csv"
    test = "D:\projects\pyprojects/fer_fmpn_pytorch\datasets/rafdb/test_ids_0.csv"
    val = "D:\projects\pyprojects/fer_fmpn_pytorch\datasets/rafdb/valid_ids_0.csv"
    save_to = "D:\projects\pyprojects/fer_fmpn_pytorch\datasets/rafdb/new/"
    # file_name = "train_ids_0.csv"
    # file_name = "test_ids_0.csv"
    # file_name = "valid_ids_0.csv"
    # remove_corrputed_images_from_splits(ptc, val, save_to, file_name)

    args.trainsplit = "train_ids_0.csv"
    args.testsplit = "test_ids_0.csv"
    args.validsplit = "valid_ids_0.csv"

    args.affectnet_img_parentfolder_man = "D:/Downloads/Manually_Annotated_compressed/"
    args.affectnet_img_parentfolder_aut = "D:/Downloads/Automatically_Annotated_compressed/"
    args.final_size = 299
    args.load_size = 299

    # train_dl, test_dl, valid_dl = get_affectnet(args=args, batch_size=8, ckp_label_type=1,remove_class=0)

    af = AffectNet(args=args, train=True, remove_class=0, ckp_label_type=1)

    af_test = AffectNet(train=False, args=args, valid=False, ckp_label_type=1, remove_class=0)
    af_valid = AffectNet(train=False, args=args, valid=True, ckp_label_type=1, remove_class=0)

    for i in range(len(af)):
        sample = af[i]
        if i % 10000 == 0:
            print("current idx: ", i)
            print(sample['image'])

    for j in range(len(af_test)):
        sample = af[j]
        if j % 1000 == 0:
            print("current idx: ", j)
            print(sample['image'])

    for k in range(len(af_valid)):
        sample = af[k]
        if k % 1000 == 0:
            print("current idx: ", k)
            print(sample['image'])





"""
    for i, batch in enumerate(train_dl):
        if i % 10000 == 0:
            print(i)
        images = batch["image"]
"""
