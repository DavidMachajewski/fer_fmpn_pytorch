import argparse
from datetime import datetime


class Setup(object):
    def __init__(self):
        super(Setup, self).__init__()

    def init(self):
        # noinspection PyTypeChecker
        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        parser.add_argument('--mode', type=str, default='train', help='Type of execution. [train|test]')
        parser.add_argument('--gpu_id', type=int, default=3, help='id of gpu to train on. Note: Check if this GPU is free before using it')
        parser.add_argument('--model_to_train', type=str, default='fmpn', help='["fmg" | "fmpn" | "resnet18" | "incv3" | "densenet | "vgg"]')
        parser.add_argument('--epochs', type=int, default=2, help="# of total epochs to train.")
        parser.add_argument('--pretrained', type=int, default=0, help="Loading the pretrained model (Default=0). [0, 1]. "
                                                                           "You need to provide path to the model.")
        parser.add_argument('--fmpn_cn', type=str, default="inc_v3", help="name of classifier model to use. ['inc_v3 | resnet18']")
        parser.add_argument('--fmpn_cn_pretrained', type=int, default=0, help="Loading imageNet pretrained cn? [0|1]")

        parser.add_argument('--dataset', type=str, default='ckp', help="[ckp|fer|affectnet]")
        parser.add_argument('--n_classes', type=int, default=7, help="# of classes. E.g. ckp has 7.")
        parser.add_argument('--data_augmentation', type=int, default=1, help="Use data augmentation for small datasets.")
        parser.add_argument('--save_samples', type=int, default=0, help="Save some images samples from training")

        # add epochs number for save_samples

        parser.add_argument('--num_workers', type=int, default=4, help="# workers for loading images parallel. Used by the dataloader.")
        parser.add_argument('--batch_size', type=int, default=8, help="batch size of dataset")

        parser.add_argument('--load_size', type=int, default=320, help='scale image to this size')
        parser.add_argument('--final_size', type=int, default=299, help='crop image to this size')

        # optimizers
        parser.add_argument('--optimizer', type=str, default="adam", help='optimizer')
        parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam optimizer')

        parser.add_argument('--lr_gen', type=float, default=0.001, help="General learning rate for trainings.")

        # schedulers
        parser.add_argument('--scheduler_type', type=str, default="linear_x", help='Scheduler for reducing lr ["linear_x" | "const"]')
        parser.add_argument('--lr_init', type=float, default=0.0001, help='initial lr at start of fmg training')
        parser.add_argument('--lr_init_after', type=float, default=0.00001, help='initial lr after reloading fmg within fmpn network')
        parser.add_argument('--lr_end', type=int, default=0, help='linear scheduler decreases to this lr')
        parser.add_argument('--start_lr_drop', type=int, default=150, help='epoch after which fmg lr is reduced')
        parser.add_argument('--start_lr_drop_fmpn', type=int, default=400, help='epoch after which fmg lr is reduced')

        # path to datasets
        parser.add_argument('--trainsplit', type=str, default="train_ids_1.csv", help="[train_0.csv|train_1.csv|...|train_9.csv]")
        parser.add_argument('--testsplit', type=str, default="test_ids_1.csv", help="[test_ids_0.csv|test_ids_1.csv|...|test_ids_9.csv]")
        parser.add_argument('--validsplit', type=str, default="valid_ids_1.csv", help="[valid_ids_0.csv|valid_ids_1.csv|...|valid_ids_9.csv]")
        parser.add_argument('--data_root', type=str, default="./datasets/", help="path to dataset folder")
        parser.add_argument('--ckp_images', type=str, default="./datasets/ckp/images_cropped/", help="path to ""cropped ckp images")
        parser.add_argument('--ckp_csvsplits', type=str, default="./datasets/ckp/tensplit/", help="path to ckp splits")
        parser.add_argument('--fer_images', type=str, default="./datasets/fer/fer2013.csv", help="path to fer dataset file")
        parser.add_argument('--fer', type=str, default="./datasets/fer/", help="path to fer dataset folder")

        parser.add_argument('--affectnet_manual', type=str, default="./datasets/affectnet/training.csv")
        parser.add_argument('--affectnet', type=str, default="./datasets/affectnet/")
        parser.add_argument('--affectnet_images', type=str, default="D:/Downloads/Manually_Annotated_compressed/Manually_Annotated_compressed/")
        #
        # :TODO: change fer_csvsplits parameter if there will not be a tensplit
        #
        parser.add_argument('--fer_csvsplits', type=str, default="./datasets/ckp/tensplit/", help="path to ckp splits")

        parser.add_argument('--masks', type=str, default="./datasets/masks/", help="path to masks folder")
        parser.add_argument('--labels', type=str, default="./datasets/masks/emotion_labels.pkl")

        # densenet params
        parser.add_argument('--dn_growth', type=int, default=32, help="growth rate of densenet")
        parser.add_argument('--dn_features', type=int, default=64, help="initial features")
        parser.add_argument('--dn_bnsize', type=int, default=4, help='')
        parser.add_argument('--norm_orig_img', type=int, default=0, help='Feeding images to pretrained Incnet req. norm.')

        # ###############################################
        # checkpoints
        # ###############################################
        parser.add_argument('--load_ckpt', type=int, default=0, help="Loading all ckpts of fmpn subnetworks"
                                                                     "to resume a training or infere data? [0|1]")
        #parser.add_argument('--ckpt_to_load', type=str,
        #                    default="./results/run_fmg_2021-04-12_18-57-23/train_fmg_2021-04-12_18-57-23\ckpt/fmg_2021-04-12_18-57-23ckpt.pth.tar",
        #                    help="")
        parser.add_argument('--ckpt_to_load', type=str, default="./results/run_fmg_2021-04-12_23-00-37/train_fmg_2021-04-12_23-00-37\ckpt/fmg_2021-04-12_23-00-37ckpt.pth.tar")

        # ckpts for fmpn
        parser.add_argument('--load_ckpt_fmg_only', type=int, default=0, help="Load just the fmg to start a first fmpn training [0|1]")
        parser.add_argument('--ckpt_fmg', type=str, default=None)
        parser.add_argument('--ckpt_cn', type=str, default=None)
        parser.add_argument('--ckpt_pfn', type=str, default=None)

        parser.add_argument('--save_ckpt_intv', type=int, default=25, help="Save training every n-th epoch.")

        # result folder
        parser.add_argument('--result_folder', type=str, default='./results/', help="")

        # deep dream model params
        parser.add_argument('--deepdream_model', type=str, default="incv3", help="")
        # parser.add_argument('--trained', type=bool, default=1, help="boolean for already trained (or not trained) FER networks")
        # evaluation
        #
        # add arguments for evaluation data dicts here
        #

        return parser

    def parse(self):
        parser = self.init()
        parser.set_defaults(name=datetime.now().strftime("%y%m%d_%H%M%S"))
        setup = parser.parse_args()
        return setup