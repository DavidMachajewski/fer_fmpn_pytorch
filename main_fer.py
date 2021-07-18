from args2 import Setup
from lib.agents.runner import Runner
from lib.dataloader.datasets import get_fer2013, FER2013
import numpy as np
import matplotlib.pyplot as plt


def fmpn_fer():
    """
    Since there are no neutral faces available which are corresponding to
    the expressive faces, the ck+ ground truth masks are used for training the fmg.
    The fmg will not be pretrained in this experiment due to the large amount
    of images inside the fer dataset.
    :return:
    """
    args = Setup().parse()

    args.mode = "train"
    args.gpu_id = 0
    args.model_to_train = "fmpn"
    args.epochs = 200
    args.fmpn_cn = "inc_v3"
    args.n_classes = 6
    args.load_ckpt = 0
    args.load_ckpt_fmg_only = 0
    args.fmg_pretrained = 0  # without pretraining
    args.fmpn_cn_pretrained = 1
    args.scheduler_type = "linear_x"
    args.dataset = "fer"
    args.save_samples = 0
    # args.dataset = "ckp"
    args.batch_size = 8
    args.load_size = 320
    args.final_size = 299
    args.save_ckpt_intv = 100
    # args.ckpt_fmg = "F:\trainings2\fmg\train_test_split\0\run_fmg_2021-06-21_14-28-02\train_fmg_2021-06-21_14-28-02\ckpt\fmg_2021-06-21_14-28-02_epoch_299_ckpt.pth.tar"
    args.trainsplit = "train_ids_0.csv"
    args.testsplit = "test_ids_0.csv"
    args.validsplit = "valid_ids_0.csv"

    runner = Runner(args)
    runner.start()


def show_ferimg():
    args = Setup().parse()
    args.trainsplit = "train_ids_0.csv"
    args.testsplit = "test_ids_0.csv"
    args.validsplit = "valid_ids_0.csv"

    fer = FER2013(args=args, train=True, ckp_label_type=True)
    fer_org = FER2013(args=args, train=True, ckp_label_type=False)

    print("label: ", fer[252]["label"])
    print("label for loading mask: ", fer[252]["label_to_load_mask"])
    plt.imshow(fer[252]["image"])
    plt.show()
    plt.imshow(fer[252]["mask"], cmap="gray")
    plt.show()


if __name__ == "__main__":
    fmpn_fer()
    # show_ferimg()
