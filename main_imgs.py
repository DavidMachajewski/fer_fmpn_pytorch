from lib.dataloader.datasets import get_ckp
from lib.dataloader.datasets import CKP, FER2013, RafDB, AffectNet
from args2 import Setup
from typing import List, Union
from argparse import ArgumentParser
from PIL import Image
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid



def print_ckp_samlpes():
    args = Setup().parse()
    args.dataset = "ckp"
    args.trainsplit = "train_ids_0.csv"
    args.testsplit = "test_ids_0.csv"
    args.validsplit = "valid_ids_0.csv"
    args.loadsize = 320
    args.finalsize = 320
    ckp = CKP(args=args, train=True)


def init_datasets(args: ArgumentParser, ds="ckp"):
    # raf = RafDB(args=args, train=True)
    # aff = AffectNet(args=args, train=True)

    if ds == "ckp":
        args.model_to_train = "fmpn"
        args.dataset = "ckp"
        args.trainsplit = "train_ids_9.csv"
        args.testsplit = "test_ids_9.csv"
        args.validsplit = "valid_ids_9.csv"
        args.ckp_label_type = 1
        args.loadsize = 320
        args.finalsize = 320
        dataset = CKP(args=args, train=True)

    elif ds == "fer":
        args.model_to_train = "fmpn"
        args.dataset = "fer"
        args.trainsplit = "train_ids_0.csv"
        args.testsplit = "test_ids_0.csv"
        args.validsplit = "valid_ids_0.csv"
        args.ckp_label_type = 0
        args.n_classes = 7
        # rgs.remove_class = 7
        args.loadsize = 320
        args.finalsize = 320
        dataset = FER2013(args=args, train=True)

    elif ds == "rafdb":
        args.model_to_train = "fmpn"
        args.dataset = "rafdb"
        args.trainsplit = "train_ids_0.csv"
        args.testsplit = "test_ids_0.csv"
        args.validsplit = "valid_ids_0.csv"
        args.ckp_label_type = 0
        args.n_classes = 7
        args.loadsize = 320
        args.finalsize = 320
        dataset = RafDB(args=args, train=True)

    elif ds == "affectnet":
        args.model_to_train = "fmpn"
        args.dataset = "affectnet"
        args.trainsplit = "train_ids_0.csv"
        args.testsplit = "test_ids_0.csv"
        args.validsplit = "valid_ids_0.csv"
        args.ckp_label_type = 0
        args.n_classes = 8
        args.loadsize = 320
        args.finalsize = 320
        args.affectnet_img_parentfolder_man = "D:\Downloads\Manually_Annotated_compressed/"
        args.affectnet_img_parentfolder_aut = "D:\Downloads\Automatically_Annotated_compressed/"
        dataset = AffectNet(args=args, train=True)

    return dataset


def search_samples_per_emotion(dataset, n_classes):
    """
    init found array
    init bilder array

    counter = 0
    Solange LÄNGE von found array nicht ANZAHL_KLASSEN
      prüfe ob für das label des aktuellen Bild im found array enthalten ist
        wenn ja
          gehe zu nächster schleife
        wenn nein
          füge label zu found array hinzu
          füge Bild zu Bilder array hinzu
      counter++
    :param dataset:
    :return:
    """

    found = []
    images = []

    counter = 16
    while len(found) < n_classes:  # n_classes
        sample = dataset[counter]
        label = sample['label']
        print(label)
        if label in found:
            counter += 4
            continue
        else:
            print("found label: ", label)
            found.append(label)
            images.append(cv.cvtColor(sample['image'], cv.COLOR_BGR2RGB))
            counter += 5
    tuples = sorted(list(zip(found, images)))
    return tuples


def create_sample_plot(args: ArgumentParser, dataset="ckp"):

    ds = init_datasets(args, dataset)
    print("classes: ", args.n_classes)

    tuples = search_samples_per_emotion(ds, args.n_classes)

    # plot images n_classes * n_classes
    fig = plt.figure(figsize=(15, 15))
    grid = ImageGrid(fig, 111, nrows_ncols=(1, args.n_classes), axes_pad=0.05)

    # append images from tuples to images
    images = [tuples[i][1] for i in range(len(tuples))]
    labels = [tuples[i][0] for i in range(len(tuples))]
    print(labels)

    for ax, im in zip(grid, images):
        ax.imshow(im)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
    plt.savefig('C:/root/uni/bachelor/bachelor_arbeit/face_expression_recognition_thesis\images\samples\datasets/affectnet_em_samples.png', bbox_inches='tight', dpi=150)
    plt.savefig('C:/root/uni/bachelor/bachelor_arbeit/face_expression_recognition_thesis\images\samples\datasets/affectnet_em_samples.pdf', bbox_inches='tight', dpi=150)
    plt.show()



def plot_histogram(path):
    img = cv.imread(path)
    cv.imshow("", img)
    plt.show()
    cv.waitKey()
    cv.imwrite("C:/root/uni/bachelor/bachelor_arbeit/face_expression_recognition_thesis\images\histograms/hist_1_mask.png", img)
    plt.hist(img.ravel(), 256, [0, 256])
    plt.yscale('log', base=10)
    plt.savefig("C:/root/uni/bachelor/bachelor_arbeit/face_expression_recognition_thesis\images\histograms/hist_1_mask_hist.png", bbox_inches='tight', dpi=300)
    plt.show()


def plot_histogram_bgr(path):
    img = cv.imread(path, cv.IMREAD_COLOR)
    cv.imshow("", img)
    plt.show()
    cv.waitKey()
    # cv.imwrite("C:/root/uni/bachelor/bachelor_arbeit/face_expression_recognition_thesis\images\histograms/hist_1_fusioned.png", img)
    # plt.hist(img.ravel(), 256, [0, 256])
    color = ('b', 'g', 'r')
    for i, col in enumerate(color):
        histr = cv.calcHist([img], [i], None, [256], [0, 256])
        plt.plot(histr, color=col)
        plt.xlim([0, 256])
    plt.savefig("C:/root/uni/bachelor/bachelor_arbeit/face_expression_recognition_thesis\images\histograms/hist_1_fusioned.png", bbox_inches='tight', dpi=300)
    plt.show()


if __name__ == "__main__":
    args = Setup().parse()
    image = "E:\david_bachelor_trainierte_netzwerke/final3107\ckp/fmpn\pretrained/run_fmpn_2021-07-30_23-40-24/train_fmpn_2021-07-30_23-40-24\plots\gray_img_1_epoch_399_batch_0.png"
    mask = "E:\david_bachelor_trainierte_netzwerke/final3107\ckp/fmpn\pretrained/run_fmpn_2021-07-30_23-40-24/train_fmpn_2021-07-30_23-40-24\plots\pred_mask_1_epoch_399_batch_0.png"
    # create_sample_plot(args=args, dataset="affectnet")
    plot_histogram(path=mask)
