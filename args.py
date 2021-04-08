import argparse
from datetime import datetime


class Setup(object):
    def __init__(self):
        super(Setup, self).__init__()

    def init(self):
        # noinspection PyTypeChecker
        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        parser.add_argument('--gpu_id', type=int, default=0, help='id of gpu to train on. Note: Check if this GPU is '
                                                                  'free before using it ')
        parser.add_argument('--mode', type=str, default='train', help='Type of execution. [train|test]')
        parser.add_argument('--model_to_train', type=str, default='fmpn', help='["fmg" | "fmpn" | "resnet18"]')
        parser.add_argument('--epochs', type=int, default=1, help="# of epochs to train.")

        parser.add_argument('--dataset', type=str, default='ckp', help="[ckp|affectnet]")

        parser.add_argument('--load_size', type=int, default=320, help='scale image to this size')
        parser.add_argument('--final_size', type=int, default=299, help='crop image to this size')

        # training settings
        parser.add_argument('--load_epoch', type=int, default=0, help='load epoch; 0: do not load')
        parser.add_argument('--niter_decay', type=int, default=100,
                            help='# of iter to linearly decay learning rate to zero')
        parser.add_argument('--niter', type=int, default=200, help='# of iter at starting learning rate')
        parser.add_argument('--epoch_count', type=int, default=1,
                            help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
        parser.add_argument('--lr_policy', type=str, default='lambda',
                            help='learning rate policy: lambda|step|plateau|cosine')
        parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate for adam')
        parser.add_argument('--lr_fmg', type=float, default=1e-5,
                            help='initial learning rate for adam for fmg.')
        parser.add_argument('--lambda_fmg', type=float, default=1e-1, help='resface weight in loss')
        parser.add_argument('--lambda_cn', type=float, default=1.0, help='discriminator weight in loss')

        # path to datasets
        parser.add_argument('--trainsplit', type=str, default="train_ids_0.csv", help="[train_0.csv|train_1.csv|...|train_9.csv]")
        parser.add_argument('--testsplit', type=str, default="test_ids_0.csv", help="[train_0.csv|train_1.csv|...|train_9.csv]")
        parser.add_argument('--data_root', type=str, default="./datasets/", help="path to dataset folder")
        parser.add_argument('--ckp_images', type=str, default="./datasets/ckp/images_cropped/", help="path to ""cropped ckp images")
        parser.add_argument('--ckp_csvsplits', type=str, default="./datasets/ckp/tensplit/", help="path to ckp splits")
        parser.add_argument('--masks', type=str, default="./datasets/masks/", help="path to masks folder")
        parser.add_argument('--labels', type=str, default="./datasets/masks/emotion_labels.pkl")

        # result folder
        parser.add_argument('--result_folder', type=str, default='./results/', help="")

        return parser

    def parse(self):
        parser = self.init()
        parser.set_defaults(name=datetime.now().strftime("%y%m%d_%H%M%S"))
        setup = parser.parse_args()
        return setup