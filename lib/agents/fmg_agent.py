import os
import time
import tqdm
import torch
import torch.nn as nn
from tqdm import trange
from lib.agents.agent import Agent
from lib.models.models import FacialMaskGenerator
from lib.dataloader.datasets import get_ckp
from lib.utils import imshow_tensor, save_tensor_img

"""
Args needed:
"""


class FmgAgent(Agent):
    def __init__(self, args):
        super(FmgAgent, self).__init__(args)
        self.name = "fmg"
        self.fmg = FacialMaskGenerator()
        self.train_dl, self.test_dl = get_ckp(args=self.args,
                                              batch_size=self.args.batch_size,
                                              shuffle=True,
                                              num_workers=0)
        self.loss = nn.MSELoss()
        self.opt = self.__init_optimizer__()

        self.device = self.__set_device__()

        # if resume training == true then load the checkpoint tar
        if self.args.load_ckpt:
            self.load_ckpt(self.args.ckpt_to_load)


    def __lambda_rule__(self, epoch):
        factor = (self.args.lr_init - self.args.lr_end) / (self.args.epochs - self.args.start_lr_drop)
        print("lr factor: ", factor)
        if epoch >= self.args.start_lr_drop:
            print("Scheduler compare: ", True)
            print("learning rate after cmp 1: ", self.opt.param_groups[0]['lr'])
            lr = self.opt.param_groups[0]['lr'] - factor
            print("learning rate after cmp 2: ", lr)
        else:
            print("scheduler compare: ", False)
            lr = self.args.lr_init
            print("learning rate after cmp: ", lr)
        return lr

    def __adjust_lr__(self):
        """Implementation of the scheduler. Adjust learning rate
        linearly to y after x epochs otherwise it is constant between 0 and y."""
        factor = (self.args.lr_init - self.args.lr_end) / (self.args.epochs - self.args.start_lr_drop)
        # print("adjusting factor: ", factor)
        if self.tmp_epoch >= self.args.start_lr_drop:
            self.opt.param_groups[0]['lr'] = self.opt.param_groups[0]['lr'] - factor
            # print("adjust lr: ", self.opt.param_groups[0]['lr'])
        else:
            self.opt.param_groups[0]['lr'] = self.args.lr_init
            # print("adjust lr < 1: ", self.opt.param_groups[0]['lr'])

    def __init_optimizer__(self):
        opt = None
        if self.args.optimizer == "adam":
            opt = torch.optim.Adam(self.fmg.parameters(),
                                   lr=self.args.lr_init,  # 0.0001
                                   betas=(self.args.beta1, 0.999))
        return opt

    def __set_device__(self):
        if self.is_cuda:
            device = torch.device(self.args.gpu_id)
            self.fmg = self.fmg.to(device)
            torch.nn.DataParallel(self.loss, device_ids=[self.args.gpu_id])
        else:
            device = torch.device("cpu")
            self.fmg = self.fmg.to(device)
        return device

    def load_ckpt(self, file_name):
        """
        :param file_name:
        :return:
        """
        print("Loading checkpoint...")
        try:
            print("Loading checkpoint {0} for {1} model".format(file_name, self.name))
            checkpoint = torch.load(file_name)

            self.tmp_epoch = checkpoint['epoch']
            self.fmg.load_state_dict(checkpoint['model_state_dict'])
            self.opt.load_state_dict(checkpoint['model_optimizer'])

            self.tmp_epoch = self.tmp_epoch
            print("\n tmp_epoch: ", self.tmp_epoch)
            print("\n tmp_lr: ", self.opt.param_groups[0]['lr'])

        except OSError as e:
            print("Could not load chkecpoint {}".format(file_name))

    def save_ckpt(self, file_name=None):
        print("Saving checkpoint...")
        if file_name is None:
            file_name = self.name + "_" + self.timestamp + "_" + "epoch_" + str(self.tmp_epoch) + "_" + "ckpt.pth.tar"
        else:
            file_name = file_name

        state = {
            'epoch': self.tmp_epoch,
            'model_state_dict': self.fmg.state_dict(),
            'model_optimizer': self.opt.state_dict()
            # 'optimizer_last_lr': self.opt.param_groups[0]['lr']
        }

        # __create_folders__() need to be launched before saving
        torch.save(state, self.train_ckpt + file_name)

    def __create_folders__(self):
        print("Creating folders...")
        super(FmgAgent, self).__create_folders__(self.name)

    def run(self):
        print("Starting fmg agent...")
        self.__create_folders__()
        self.save_args(self.run_path + "args.txt")
        #
        # train / test / both
        #
        self.train()
        self.test()

    def train(self):
        if self.args.load_ckpt:
            print("Resuming training...")
        else:
            print("Starting training...")

        self.fmg.train()

        with trange(self.tmp_epoch, self.args.epochs, desc="Epoch", unit="epoch") as epochs:
            for epoch in epochs:
                self.tmp_epoch = epoch
                epoch_loss, pred_mask_sample, sample_label = self.train_epoch()

                self.list_train_loss.append(sample_label.cpu().detach().numpy())
                self.list_lr.append(self.opt.param_groups[0]['lr'])

                if epoch % 5 == 0:
                    save_tensor_img(img=pred_mask_sample.cpu().detach(),
                                    path=self.train_plots + "mask_" + str(
                                        sample_label.cpu().detach().numpy()) + "_epoch_" + str(epoch) + ".png")

                self.__adjust_lr__()

                epochs.set_postfix(loss="{:.3f}".format(epoch_loss, prec='.3'),
                                   lr="{:.8f}".format(self.opt.param_groups[0]['lr'], prec='.8'))

                if epoch % self.args.save_ckpt_intv == 0:
                    self.save_ckpt()
                    self.save_resultlists_as_dict(self.train_path + "/" + "epoch_" + str(epoch) + "_train_logs.pickle")

        self.save_resultlists_as_dict(self.train_path + "end_train_logs.pickle")
        self.save_ckpt()

        # optionally saving of checkpoints while training. Make argument e.g. self.args.save_nth_ckpt = 10

    def train_epoch(self):
        """
        :return: Epoch loss and a sample of a predicted mask
        """
        epoch_loss, predicted_masks = 0.0, None
        with tqdm.tqdm(self.train_dl, desc="Batch", unit="batches") as tbatch:
            for i, batch in enumerate(tbatch):
                images_gray = batch["image_gray"].to(self.device)
                label_masks = batch["mask"].to(self.device)
                labels = batch["label"]

                self.opt.zero_grad()

                predicted_masks = self.fmg(images_gray).to(self.device)
                tmp_loss = self.loss(predicted_masks, label_masks)

                tmp_loss.backward()
                self.opt.step()

                epoch_loss += tmp_loss.item()
            epoch_loss = epoch_loss / len(self.train_dl)  # get the real sample number from dataset and batch size
        return epoch_loss, predicted_masks[0], labels[0]  # we dont need accuracy for fmg training

    def test(self, path=None):
        """Predicts masks for the dataset"""
        # 2 cases
        #
        # 1. directly after training
        # 2. reload trained fmg and predict masks
        #
        print("Starting evaluation...")
        self.fmg.eval()

        if path is not None:
            save_to = path
        else:
            save_to = self.test_plots

        for i, batch in enumerate(self.test_dl):
            image_org = batch["image"]
            images_gray = batch["image_gray"].to(self.device)
            label_masks = batch["mask"].to(self.device)
            labels = batch["label"]
            #
            #
            # :TODO: Create some images for test set as well to compare train and test
            #
            predicted_masks = self.fmg(images_gray).to(self.device)
            save_tensor_img(img=images_gray[0].cpu().detach(),
                            path=save_to + "orig_gray_img_"
                                 + str(labels[0].cpu().detach().numpy()) + "_batch_" + str(i) + ".png")
            save_tensor_img(img=predicted_masks[0].cpu().detach(),
                            path=save_to + "pred_mask_"
                                 + str(labels[0].cpu().detach().numpy()) + "_batch_" + str(i) + ".png")
            imshow_tensor(img=predicted_masks[0].cpu().detach(),
                            path=save_to + "pred_mask_"
                                 + str(labels[0].cpu().detach().numpy()) + "_batch_" + str(i) + "_heat.png",
                          one_channel=True)
            save_tensor_img(img=label_masks[0].cpu().detach(),
                            path=save_to + "orig_mask_"
                                 + str(labels[0].cpu().detach().numpy()) + "_batch_" + str(i) + ".png")
            imshow_tensor(img=predicted_masks[0].cpu().detach(),
                          path=save_to + "orig_mask_"
                               + str(labels[0].cpu().detach().numpy()) + "_batch_" + str(i) + "_heat.png",
                          one_channel=True)


def get_scheduler(optimizer, args):
    if args.scheduler_type == 'linear':
        def lambda_rule(epoch):
            factor = (args.lr_init - args.lr_end) / (args.epochs - args.start_lr_drop)
            print("lr factor: ", factor)
            if epoch >= args.start_lr_drop:
                print("Scheduler compare: ", True)
                print("learning rate after cmp 1: ", optimizer.param_groups[0]['lr'])
                lr = optimizer.param_groups[0]['lr'] - factor
                print("learning rate after cmp 2: ", lr)
            else:
                print("scheduler compare: ", False)
                lr = args.lr_init
                print("learning rate after cmp: ", lr)
            return lr

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    return scheduler
