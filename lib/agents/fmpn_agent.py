from lib.agents.agent import Agent
from lib.models.models import FacialMaskGenerator, PriorFusionNetwork
from lib.dataloader.datasets import get_ckp
import torch.nn as nn
import torch
from tqdm import trange, tqdm
import torchvision as tv
import pickle


class FmpnAgent(Agent):
    def __init__(self, args):
        super(FmpnAgent, self).__init__(args)
        self.name = "fmpn"
        self.fmg = FacialMaskGenerator()
        self.pfn = PriorFusionNetwork()
        # self.cn = tv.models.Inception3(num_classes=7, init_weights=True)
        self.cn = torch.hub.load('pytorch/vision:v0.9.0', 'inception_v3', pretrained=True)

        self.opt = self.__init_optimizer__()

        self.device = self.__set_device__()
        self.fmg.to(self.device)
        self.pfn.to(self.device)
        self.cn.to(self.device)

        self.train_dl, self.test_dl = get_ckp(args=self.args,
                                              batch_size=self.args.batch_size,
                                              shuffle=True,
                                              num_workers=4)
        self.loss_fmg_fn = nn.MSELoss()
        self.loss_cn_fn = nn.CrossEntropyLoss()

        # self.device = self.__set_device__()

        # self.opt = self.__init_optimizer__()

        self.list_train_loss_fmg = []
        self.list_train_loss_cn = []
        self.list_train_loss_cn = []
        self.list_lr_fmg = []
        self.list_lr_cn = []
        self.list_lr_pfn = []

        self.epoch_counter = 0

        if self.args.load_ckpt:
            self.opt.add_param_group({'params': self.pfn.parameters()})
            self.opt.add_param_group({'params': self.cn.parameters()})
            self.load_ckpt(file_name_fmg=self.args.ckpt_fmg,  # load the pretrained fmg
                           file_name_pfn=self.args.ckpt_pfn,
                           file_name_cn=self.args.ckpt_cn)

        if self.args.load_ckpt_fmg_only:
            self.load_fmg(file_name_fmg=self.args.ckpt_fmg)
            self.opt.add_param_group({'params': self.pfn.parameters()})
            self.opt.add_param_group({'params': self.cn.parameters()})




    def loss_total_fn(self, loss_fmg, loss_cn):
        """10, 1 are lambda1, lambda2 - add to args!!!"""
        return 10*loss_fmg + 1*loss_cn  # lambda1, lambda2 to args!

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
        factor_cn = (self.args.lr_init - self.args.lr_end) / (self.args.epochs - self.args.start_lr_drop_fmpn)
        factor_fmg = (self.args.lr_init_after - self.args.lr_end) / (self.args.epochs - self.args.start_lr_drop_fmpn)
        print("adjusting factor cn: ", factor_cn)
        print("adjusting factor fmg: ", factor_fmg)
        if self.tmp_epoch >= self.args.start_lr_drop_fmpn:  # >= 400
            # we load fmg with tmp_epoch value of 300
            # for group in self.opt.param_groups:
            #     group['lr'] = group['lr'] - factor
            #     print("adjust lrts: ", group['lr'])
            self.opt.param_groups[0]['lr'] = self.opt.param_groups[0]['lr'] - factor_fmg
            self.opt.param_groups[1]['lr'] = self.opt.param_groups[1]['lr'] - factor_cn
            self.opt.param_groups[2]['lr'] = self.opt.param_groups[2]['lr'] - factor_cn
        else:
            # keep fmg lr at 0.00001 until epoch 400
            self.opt.param_groups[0]['lr'] = self.args.lr_init_after
            # lr for pfn, cn stay constant at 0.0001 until epoch 400
            self.opt.param_groups[1]['lr'] = self.args.lr_init
            self.opt.param_groups[2]['lr'] = self.args.lr_init

    def __init_optimizer__(self):
        # opt = None
        if self.args.optimizer == "adam":
            """
            opt = torch.optim.Adam(
                [
                    {'params': self.fmg.parameters()},
                    {'params': self.pfn.parameters()},
                    {'params': self.cn.parameters()}
                ],
                lr=self.args.lr_init,
                betas=(self.args.beta1, 0.999)
            )
            """
            opt = torch.optim.Adam(self.fmg.parameters(),lr=self.args.lr_init,betas=(self.args.beta1, 0.9999))
        return opt

    def __set_device__(self):
        if self.is_cuda:
            device = torch.device(self.args.gpu_id)
        else:
            device = torch.device("cpu")
        return device

    def load_fmg(self, file_name_fmg):
        """Use this function for first time training the
        fmpn to load the pretrained fmg model."""
        try:
            fmg_ckpt = torch.load(file_name_fmg)
            self.tmp_epoch = fmg_ckpt['epoch']
            self.fmg.load_state_dict(fmg_ckpt['model_state_dict'])
            self.opt.load_state_dict(fmg_ckpt['model_optimizer'])
            self.opt.param_groups[0]['lr'] = self.args.lr_init_after
        except OSError as e:
            print("Could not load ckpt.")

    def load_ckpt(self, file_name_fmg, file_name_pfn, file_name_cn):
        try:
            print(file_name_fmg)
            print(file_name_pfn)
            print(file_name_cn)
            fmg_ckpt = torch.load(file_name_fmg)
            pfn_ckpt = torch.load(file_name_pfn)
            cn_ckpt = torch.load(file_name_cn)

            self.tmp_epoch = fmg_ckpt['epoch']

            self.fmg.load_state_dict(fmg_ckpt['model_state_dict'])
            self.pfn.load_state_dict(pfn_ckpt['model_state_dict'])
            self.cn.load_state_dict(cn_ckpt['model_state_dict'])

            self.opt.param_groups[0] = fmg_ckpt['model_optimizer']
            self.opt.param_groups[1] = pfn_ckpt['model_optimizer']
            self.opt.param_groups[2] = cn_ckpt['model_optimizer']

            # self.tmp_epoch = self.tmp_epoch
            self.epoch_counter = cn_ckpt['epoch']

        except OSError as e:
            print("Could not load checkpoints.")

    def save_ckpt(self):

        file_name_fmg = self.name + "_fmg_" + self.timestamp + "_" + "epoch_" + str(self.tmp_epoch) + "_" "ckpt.pth.tar"
        file_name_pfn = self.name + "_pfn_" + self.timestamp + "_" + "epoch_" + str(self.tmp_epoch) + "_" "ckpt.pth.tar"
        file_name_cn = self.name + "_cn_" + self.timestamp + "_" + "epoch_" + str(self.tmp_epoch) + "_" "ckpt.pth.tar"

        state_fmg = {
            'epoch': self.tmp_epoch,
            'model_state_dict': self.fmg.state_dict(),
            'model_optimizer': self.opt.param_groups[0],  # dict
        }
        state_pfn = {
            'epoch': self.epoch_counter,  # create counter
            'model_state_dict': self.pfn.state_dict(),
            'model_optimizer': self.opt.param_groups[1],
        }
        state_cn = {
            'epoch': self.epoch_counter,
            'model_state_dict': self.cn.state_dict(),
            'model_optimizer': self.opt.param_groups[2],
            'cn_name': self.args.fmpn_cn
        }
        torch.save(state_fmg, self.train_ckpt + file_name_fmg)
        torch.save(state_pfn, self.train_ckpt + file_name_pfn)
        torch.save(state_cn, self.train_ckpt + file_name_cn)

    def __create_folders__(self):
        super(FmpnAgent, self).__create_folders__(self.name)

    def save_resultlists_as_dict(self, path):
        dict = {
            'train_loss': self.list_train_loss,
            'train_loss_fmg': self.list_train_loss_fmg,
            'train_loss_cn': self.list_train_loss_cn,
            'train_acc': self.list_train_acc,
            'test_loss': self.list_test_loss,
            'test_acc': self.list_test_acc,
            'lr_fmg': self.list_lr_fmg,
            'lr_pfn': self.list_lr_pfn,
            'lr_cn': self.list_lr_cn
        }
        file = open(path, "wb")
        pickle.dump(dict, file)

    def run(self):
        self.__create_folders__()
        self.save_args(self.run_path + "args.txt")
        self.train()
        self.test()

    def train(self):
        self.fmg.train()
        self.pfn.train()
        self.cn.train()

        with trange(self.tmp_epoch, self.args.epochs, desc="Epoch", unit="epoch") as epochs:
            for epoch in epochs:
                self.tmp_epoch = epoch
                self.epoch_counter += 1

                # train one epoch
                epoch_loss, epoch_acc, epoch_loss_fmg, epoch_loss_cn = self.train_epoch()

                # append learning rates for fmg and cn and total
                self.list_train_loss.append(epoch_loss)
                self.list_train_loss_fmg.append(epoch_loss_fmg)
                self.list_train_loss_cn.append(epoch_loss_cn)

                self.list_lr_fmg.append(self.opt.param_groups[0]['lr'])
                self.list_lr_pfn.append(self.opt.param_groups[1]['lr'])
                self.list_lr_cn.append(self.opt.param_groups[2]['lr'])

                self.list_train_acc.append(epoch_acc)

                self.__adjust_lr__()

                epochs.set_postfix(loss="{:.3f}".format(epoch_loss, prec='.3'),
                                   accuracy="{:.3f}".format(epoch_acc.item(), prec='.3'),
                                   lr_fmg="{:.6f}".format(self.list_lr_fmg[-1], prec='.6'),
                                   lr_cn="{:.6f}".format(self.list_lr_cn[-1], prec='.6'))

                if epoch % 25 == 0:
                    self.save_ckpt()
                    self.save_resultlists_as_dict(self.train_path + "/" + "epoch_" + str(epoch) + "_train_logs.pickle")

            self.save_resultlists_as_dict(self.train_path + "end_train_logs.pickle")
            # save last ckpt
            self.save_ckpt()


    def train_epoch(self):
        epoch_loss = 0.0
        epoch_fmg_loss = 0.0
        epoch_cn_loss = 0.0
        epoch_acc = 0.0
        with tqdm(self.train_dl, desc="Batch", unit="batches") as tbatch:
            for i, batch in enumerate(tbatch):
                images = batch["image"].to(self.device)
                images_gray = batch["image_gray"].to(self.device)
                label_masks = batch["mask"].to(self.device)
                labels = batch["label"].to(self.device)

                self.opt.zero_grad()

                predicted_masks = self.fmg(images_gray).to(self.device)
                heat_face = images_gray * predicted_masks
                heat_face.to(self.device)
                fusion_img = self.pfn(images, heat_face)
                classifications = self.cn(fusion_img)

                fmg_loss = self.loss_fmg_fn(predicted_masks, label_masks)
                cn_loss = self.loss_cn_fn(classifications.logits, labels)
                epoch_fmg_loss += fmg_loss.item()
                epoch_cn_loss += cn_loss.item()

                total_loss = self.loss_total_fn(fmg_loss, cn_loss)

                total_loss.backward()
                self.opt.step()

                epoch_loss += total_loss.item()
                epoch_acc += self.calc_accuracy(classifications.logits, labels)
            epoch_acc = epoch_acc / len(self.train_dl)

            epoch_fmg_loss = epoch_fmg_loss / len(self.train_dl)
            epoch_cn_loss = epoch_cn_loss / len(self.train_dl)

            epoch_loss = epoch_loss / len(self.train_dl)
        return epoch_loss, epoch_acc, epoch_fmg_loss, epoch_cn_loss

    def calc_accuracy(self, predictions, labels):
        # print("Predictions: \n", predictions)
        # print("Labels: \n", labels)
        classes = torch.argmax(predictions, dim=-1)
        # print("Classes after argmax: \n", classes)
        return torch.mean((classes == labels).float())

    def test(self):
        self.fmg.eval()
        self.pfn.eval()
        self.cn.eval()

        epoch_fmg_val_loss = 0.0
        epoch_cn_val_loss = 0.0
        with tqdm(self.test_dl, desc="Batch", unit="batches") as tbatch:
            for i, batch in enumerate(tbatch):
                images = batch["image"].to(self.device)
                images_gray = batch["image_gray"].to(self.device)
                label_masks = batch["mask"].to(self.device)
                labels = batch["label"].to(self.device)

                predicted_masks = self.fmg(images_gray).to(self.device)
                heat_face = images_gray * predicted_masks
                heat_face.to(self.device)
                fusion_img = self.pfn(images, heat_face)
                classifications = self.cn(fusion_img)

                fmg_loss = self.loss_fmg_fn(predicted_masks, label_masks)
                cn_loss = self.loss_cn_fn(classifications.logits, labels)
                epoch_fmg_val_loss += fmg_loss.item()
                epoch_cn_val_loss += cn_loss.item()

                total_loss = self.loss_total_fn(fmg_loss, cn_loss)



