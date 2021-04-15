import argparse
from datetime import datetime


class Setup(object):
    def __init__(self):
        super(Setup, self).__init__()

    def init(self):
        # noinspection PyTypeChecker
        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        parser.add_argument('--gpu_id', type=int, default=0, help='id of gpu to train on. Note: Check if this GPU is free before using it')
        parser.add_argument('--mode', type=str, default='train', help='Type of execution. [train|test]')
        parser.add_argument('--model_to_train', type=str, default='fmpn', help='["fmg" | "fmpn" | "resnet18"]')
        parser.add_argument('--epochs', type=int, default=2, help="# of total epochs to train.")
        parser.add_argument('--fmpn_cn', type=str, default="inc_v3", help="name of classifier model to use. ['inc_v3 | resnet18']")

        parser.add_argument('--dataset', type=str, default='ckp', help="[ckp|affectnet]")
        parser.add_argument('--batch_size', type=int, default=8, help="batch size of dataset")

        parser.add_argument('--load_size', type=int, default=320, help='scale image to this size')
        parser.add_argument('--final_size', type=int, default=299, help='crop image to this size')

        # optimizers
        parser.add_argument('--optimizer', type=str, default="adam", help='optimizer')
        parser.add_argument('--beta1', type=int, default=0.5, help='beta1 for Adam optimizer')

        # schedulers
        parser.add_argument('--scheduler_type', type=str, default="linear", help='Scheduler for reducing lr ["linear"]')
        parser.add_argument('--lr_init', type=int, default=0.0001, help='initial lr at start of fmg training')
        parser.add_argument('--lr_init_after', type=int, default=0.00001, help='initial lr after reloading fmg within fmpn network')
        parser.add_argument('--lr_end', type=int, default=0, help='linear scheduler decreases to this lr')
        parser.add_argument('--start_lr_drop', type=int, default=150, help='epoch after which fmg lr is reduced')
        parser.add_argument('--start_lr_drop_fmpn', type=int, default=400, help='epoch after which fmg lr is reduced')

        # path to datasets
        parser.add_argument('--trainsplit', type=str, default="train_ids_0.csv", help="[train_0.csv|train_1.csv|...|train_9.csv]")
        parser.add_argument('--testsplit', type=str, default="test_ids_0.csv", help="[train_0.csv|train_1.csv|...|train_9.csv]")
        parser.add_argument('--data_root', type=str, default="./datasets/", help="path to dataset folder")
        parser.add_argument('--ckp_images', type=str, default="./datasets/ckp/images_cropped/", help="path to ""cropped ckp images")
        parser.add_argument('--ckp_csvsplits', type=str, default="./datasets/ckp/tensplit/", help="path to ckp splits")
        parser.add_argument('--masks', type=str, default="./datasets/masks/", help="path to masks folder")
        parser.add_argument('--labels', type=str, default="./datasets/masks/emotion_labels.pkl")

        #
        # checkpoints
        #
        parser.add_argument('--load_ckpt', type=bool, default=False)
        #parser.add_argument('--ckpt_to_load', type=str,
        #                    default="./results/run_fmg_2021-04-12_18-57-23/train_fmg_2021-04-12_18-57-23\ckpt/fmg_2021-04-12_18-57-23ckpt.pth.tar",
        #                    help="")
        parser.add_argument('--ckpt_to_load', type=str, default="./results/run_fmg_2021-04-12_23-00-37/train_fmg_2021-04-12_23-00-37\ckpt/fmg_2021-04-12_23-00-37ckpt.pth.tar")

        # ckpts for fmpn
        parser.add_argument('--load_ckpt_fmg_only', type=bool, default=False)
        parser.add_argument('--ckpt_fmg', type=str, default=None)
        parser.add_argument('--ckpt_cn', type=str, default=None)
        parser.add_argument('--ckpt_pfn', type=str, default=None)

        # result folder
        parser.add_argument('--result_folder', type=str, default='./results/', help="")

        return parser

    def parse(self):
        parser = self.init()
        parser.set_defaults(name=datetime.now().strftime("%y%m%d_%H%M%S"))
        setup = parser.parse_args()
        return setup