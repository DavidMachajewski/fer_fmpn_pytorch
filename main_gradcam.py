from args2 import Setup
from lib.agents.fmpn_agent import FmpnAgent
from lib.agents.inc_agent import InceptionAgent
from lib.featurevisualization.grad_cam import GradCAMAgent


def set_args(model_name):
    if model_name == "fmpn":
        args = Setup().parse()

        args.model_to_train = "fmpn"
        args.mode = "test"
        args.gpu_id = 0
        args.trainsplit = "train_ids_8.csv"
        args.testsplit = "test_ids_8.csv"
        args.validsplit = "valid_ids_8.csv"
        args.load_ckpt = 1
        args.batch_size = 8
        args.load_size = 299
        args.final_size = 299
        args.augmentation = 0
        args.fmpn_cn_pretrained = 1
        args.ckpt_fmg = "E:/david_bachelor_trainierte_netzwerke/final3107/ckp/fmpn/pretrained/run_fmpn_2021-07-30_23-41-06/train_fmpn_2021-07-30_23-41-06/ckpt/fmpn_fmg_2021-07-30_23-41-06_epoch_498_ckpt.pth.tar"
        args.ckpt_pfn = "E:/david_bachelor_trainierte_netzwerke/final3107/ckp/fmpn/pretrained/run_fmpn_2021-07-30_23-41-06/train_fmpn_2021-07-30_23-41-06/ckpt/fmpn_pfn_2021-07-30_23-41-06_epoch_498_ckpt.pth.tar"
        args.ckpt_cn = "E:/david_bachelor_trainierte_netzwerke/final3107/ckp/fmpn/pretrained/run_fmpn_2021-07-30_23-41-06/train_fmpn_2021-07-30_23-41-06/ckpt/fmpn_cn_2021-07-30_23-41-06_epoch_498_ckpt.pth.tar"
        return args
    elif model_name == "incv3":
        args = Setup().parse()

        args.model_to_train = "incv3"
        args.mode = "test"
        args.gpu_id = 0
        args.trainsplit = "train_ids_8.csv"
        args.testsplit = "test_ids_8.csv"
        args.validsplit = "valid_ids_8.csv"
        args.load_size = 299
        args.final_size = 299
        args.pretrained = 1
        args.load_ckpt = 1
        args.batch_size = 8
        args.ckpt_to_load = "E:/david_bachelor_trainierte_netzwerke/final3107/ckp/inceptionnet/pretrained/run_incv3_2021-07-31_21-41-30/train_incv3_2021-07-31_21-41-30/ckpt/incv3_epoch_199_ckpt.pth.tar"
        return args


if __name__ == "__main__":
    # args = set_args("fmpn")
    args = set_args("incv3")

    # agent = FmpnAgent(args)
    agent = InceptionAgent(args)

    # layers = [-4, -5, -6, -8, -9, -10, -11, -12, -13, -14]
    layers = [-15, -16]
    # layers = [-15, -16]
    for layer in layers:
        gradcam = GradCAMAgent(agent, target_layer_nr=layer)
        gradcam.create_cams(save_to="C:/root/uni/bachelor/gradcams/")
        del gradcam
