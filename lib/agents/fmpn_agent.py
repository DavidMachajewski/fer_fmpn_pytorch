from lib.agents.agent import Agent
from lib.models.models import FacialMaskGenerator, PriorFusionNetwork
from lib.dataloader.datasets import get_ckp
import torch.nn as nn
import torch
from tqdm import trange

class FmpnAgent(Agent):
    def __init__(self, args):
        super(FmpnAgent, self).__init__(args)
        self.name = "fmpn"
        self.fmg = FacialMaskGenerator()
        self.pfn = PriorFusionNetwork()
        # self.cn

        self.train_dl, self.test_dl = get_ckp(args=self.args,
                                              batch_size=self.args.batch_size,
                                              shuffle=True,
                                              num_workers=4)
        self.loss_fmg_fn = nn.MSELoss()
        self.loss_cn_fn = nn.CrossEntropyLoss()

        self.opt = self.__init_optimizer__()

        self.device = self.__set_device__()

        if self.args.load_ckpt:
            self.load_ckpt(self.args.ckpt_to_load)

    def loss_total_fn(self, loss_fmg, loss_cn):
        """10, 1 are lambda1, lambda2 - add to args!!!"""
        return 10*loss_fmg + 1*loss_cn

    def __adjust_lr__(self):
        """Adjust the learning rate.
        1) Training starts by tuning only FMG for 300
        epochs using adam. The lr is decayed linearly
        to 0 from epoch 150.
        2) After that train the whole
        fmpn for 200 epochs jointly and linearly decay the
        lr from epoch 100. But first reset learning rate
        of fmg to 0.00001 while rest uses 0.0001.
        Step 1) is done by first run of the fmg
        trained by a single run."""
        factor = (self.args.lr_init - self.args.lr_end) / (self.args.epochs - self.args.start_lr_drop_fmpn)
        print("adjusting factor: ", factor)
        if self.tmp_epoch >= self.args.start_lr_drop_fmpn:  # 400
            # we load fmg with tmp_epoch value of 300
            for group in self.opt.param_groups:
                group['lr'] = group['lr'] - factor
                print("adjust lrts: ", group['lr'])
        else:
            # keep fmg lr at 0.00001 until epoch 400
            self.opt.param_groups[0]['lr'] = self.args.lr_init_after
            # lr for pfn, cn stay constant at 0.0001 until epoch 400
            self.opt.param_groups[1]['lr'] = self.args.lr_init
            self.opt.param_groups[2]['lr'] = self.args.lr_init


    def __init_optimizer__(self):
        opt = None
        if self.args.optimizer == "adam":
            opt = torch.optim.Adam(
                params=
                [
                    {'params': self.fmg.parameters()},
                    {'params': self.pfn.parameters()},
                    {'params': self.cn.parameters()}
                ],
                lr=self.args.lr,
                betas=(self.args.beta1, 0.999)
            )
        return opt

    def __set_device__(self):
        if self.is_cuda:
            device = torch.device(self.args.gpu_id)
        else:
            device = torch.device("cpu")
        self.fmg.to(device)
        self.pfn.to(device)
        self.cn.to(device)
        return device


    def load_ckpt(self, file_name_fmg, file_name_pfn, file_name_cn):
        #
        # restore models -> FMG, PFN, CN
        #
        # when loaded checkpoints, directly reset
        # the learning rate of fmg to 0.00001
        # and the other learning rates to 0.0001
        try:
            fmg_ckpt = torch.load(file_name_fmg)
            pfn_ckpt = torch.load(file_name_pfn)
            cn_ckpt = torch.load(file_name_cn)

            self.tmp_epoch = fmg_ckpt['epoch']

            self.fmg.load_state_dict(fmg_ckpt['model_state_dict'])
            self.pfn.load_state_dict(pfn_ckpt['model_state_dict'])
            self.cn.load_state_dict(cn_ckpt['model_state_dict'])

            self.opt.param_groups[0] = fmg_ckpt['model_optimizer_dict']
            self.opt.param_groups[1] = pfn_ckpt['model_optimizer_dict']
            self.opt.param_groups[2] = cn_ckpt['model_optimizer_dict']

            self.tmp_epoch = self.tmp_epoch[0]

        except OSError as e:
            print("Could not load checkpoints.")
        pass

    def save_ckpt(self):

        file_name_fmg = self.name + "_fmg_" + self.timestamp + "ckpt.pth.tar"
        file_name_pfn = self.name + "_pfn_" + self.timestamp + "ckpt.pth.tar"
        file_name_cn = self.name + "_cn_" + self.timestamp + "ckpt.pth.tar"

        state_fmg = {
            'epoch': self.tmp_epoch,
            'model_state_dict': self.fmg.state_dict(),
            'model_optimizer_dict': self.opt.param_groups[0],  # dict
        }
        state_pfn = {
            'epoch': self.tmp_epoch,
            'model_state_dict': self.pfn.state_dict(),
            'model_optimizer_dict': self.opt.param_groups[1],
        }
        state_cn = {
            'epoch': self.tmp_epoch,
            'model_state_dict': self.cn.state_dict(),
            'model_optimizer_dict': self.opt.param_groups[2],
            'cn_name': self.args.fmpn_cn
        }
        torch.save(state_fmg, self.train_ckpt + file_name_fmg)
        torch.save(state_pfn, self.train_ckpt + file_name_pfn)
        torch.save(state_cn, self.train_ckpt + file_name_cn)

    def __create_folders__(self):
        super(FmpnAgent, self).__create_folders__(self.name)

    def run(self):
        self.__create_folders__()
        self.save_args(self.run_path + "args.txt")
        self.train()
        self.test()

    def train(self):
        with trange(self.tmp_epoch, self.args.epochs, desc="Epoch", unit="epoch") as epochs:
            for epoch in epochs:
                self.tmp_epoch = epoch
                epoch_loss, pred_mask_sample, sample_label = self.train_epoch()
                #
                #
                #

    def train_epoch(self):
        self.fmg.train()
        self.pfn.train()
        self.cn.train()
        #
        #
        #

    def test(self):
        pass