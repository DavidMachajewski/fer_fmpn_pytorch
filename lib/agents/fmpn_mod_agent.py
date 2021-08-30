import torch
import pickle
import torch.nn as nn
import numpy as np
from tqdm import trange, tqdm
from lib.utils import save_tensor_img
from lib.agents.agent import Agent
from lib.dataloader.datasets import get_ckp, get_fer2013, get_rafdb
from lib.eval.eval_utils import make_cnfmat_plot, prec_recall_fscore, roc_auc_score
from lib.models.models import FacialMaskGenerator, PriorFusionNetwork, inceptionv3, SCNN0
from torchsummary import summary


#################
# RUN WITH PRETRAINED InceptionNet as classification network
# >> for %i in (0) do python main.py --mode train --gpu_id 0 --model_to_train fmpn --epochs 500 --save_ckpt_intv 50 --load_size 320 --final_size 299 --load_ckpt_fmg_only 1 --fmpn_cn inc_v3 --fmpn_cn_pretrained 1 --dataset ckp --batch_size 8 --scheduler_type linear_x --ckpt_fmg ./results/run_fmg_2021-04-14_13-57-45/train_fmg_2021-04-14_13-57-45\ckpt/fmg_ckpt.pth.tar --trainsplit train_ids_%i.csv --testsplit test_ids_%i.csv
#
# for %i in (0) do python main.py --mode train --gpu_id 0 --model_to_train fmpn --epochs 500 --save_ckpt_intv 50 --load_size 320 --final_size 299 --load_ckpt_fmg_only 1 --fmpn_cn
# inc_v3 --fmpn_cn_pretrained 1 --dataset ckp --batch_size 8 --scheduler_type linear_x --ckpt_fmg ./results/run_fmg_2021-04-14_13-57-45/train_fmg_2021-04-14_13-57-45\ckpt/fmg_ckpt.pth.tar --trainsplit train_ids_%i.csv --testsplit test_i
# ds_%i.csv
#################


class FmpnAgentMod(Agent):
    def __init__(self, args):
        super(FmpnAgentMod, self).__init__(args)
        self.name = self.args.model_to_train
        self.fmg = FacialMaskGenerator()
        #
        # PFN NOT NEEDED HERE.
        #
        self.pfn = PriorFusionNetwork()

        # ADD A CONV LAYER TO THE FRONT OF INCEPTION NET
        # TO CONVOLVE FROM 4 TO 3 DIMS WITHOUT CHANGING
        # THE IMAGE SIZE
        self.cn = self.init_cn()

        if not self.args.cls_masks:
            self.cn_modinp = torch.nn.Conv2d(in_channels=4, out_channels=3, kernel_size=(3, 3), padding=(1, 1))
            self.cn = nn.Sequential(*[self.cn_modinp, self.cn])



        self.opt = self.__init_optimizer__()

        self.device = self.__set_device__()
        torch.cuda.set_device(self.device)

        self.fmg = self.fmg.cuda()
        self.pfn = self.pfn.cuda()
        self.cn = self.cn.cuda()

        print("fmg, pnf, cn cuda? {0}, {1}, {2}".format(
            next(self.fmg.parameters()).is_cuda,
            next(self.pfn.parameters()).is_cuda,
            next(self.cn.parameters()).is_cuda)
        )

        # :TODO: LOADING DATALOADER CAN BE DONE BY EXTRA FUNCTION
        if self.args.dataset == "ckp":
            self.train_dl, self.test_dl, self.valid_dl = get_ckp(args=self.args,
                                                                 batch_size=self.args.batch_size,
                                                                 shuffle=True,
                                                                 num_workers=self.args.num_workers,
                                                                 drop_last=True,
                                                                 valid=True)
            print("Loaded ckp dataset")
        elif self.args.dataset == "fer":
            self.train_dl, self.test_dl, self.valid_dl = get_fer2013(args=self.args,
                                                                     batch_size=self.args.batch_size,
                                                                     shuffle=True,
                                                                     num_workers=self.args.num_workers,
                                                                     drop_last=True,
                                                                     augmentation=self.args.augmentation,
                                                                     remove_class=self.args.remove_class,
                                                                     ckp_label_type=True)
            print("Loaded fer dataset")
        elif self.args.dataset == "rafdb":
            self.train_dl, self.test_dl, self.valid_dl = get_rafdb(args=self.args,
                                                                   ckp_label_type=self.args.ckp_label_type,
                                                                   batch_size=self.args.batch_size,
                                                                   shuffle=True,
                                                                   num_workers=self.args.num_workers,
                                                                   drop_last=True,
                                                                   augmentation=self.args.augmentation,
                                                                   remove_class=self.args.remove_class)  # remove neutral class
            print("Loaded rafdb dataset")
        self.loss_fmg_fn = nn.MSELoss()
        self.loss_cn_fn = nn.CrossEntropyLoss()

        self.list_train_loss_fmg = []
        self.list_train_loss_pfn = []
        self.list_train_loss_cn = []
        self.list_test_loss_fmg = []
        self.list_test_loss_pfn = []
        self.list_test_loss_cn = []

        self.list_lr_fmg = []
        self.list_lr_cn = []
        self.list_lr_pfn = []

        self.epoch_counter = 0

        self.train_logs_path = None

        if self.args.load_ckpt:
            # resume already trained fmpn by loading the ckpts for fmg, pfn und cn
            # use this to continue training or infere images
            self.opt.add_param_group({'params': self.pfn.parameters()})
            self.opt.add_param_group({'params': self.cn.parameters()})
            self.load_ckpt(file_name_fmg=self.args.ckpt_fmg,  # load the pretrained fmg
                           file_name_pfn=self.args.ckpt_pfn,
                           file_name_cn=self.args.ckpt_cn)

        elif self.args.load_ckpt_fmg_only:
            # resume just the facial mask generator. Use this for first fmpn trainings
            self.load_fmg(file_name_fmg=self.args.ckpt_fmg)
            self.opt.add_param_group({'params': self.pfn.parameters()})
            self.opt.add_param_group({'params': self.cn.parameters()})
        elif not self.args.fmg_pretrained:
            print("Training just the second step of fmpn")
            # train without pretraining the fmg. Use this for big datasets like fer or affectnet
            self.opt.param_groups[0]['lr'] = self.args.lr_init_after
            self.opt.add_param_group({'params': self.pfn.parameters()})
            self.opt.add_param_group({'params': self.cn.parameters()})


    def init_cn(self):
        print("Initializing classification network...")
        if self.args.fmpn_cn == "inc_v3":
            return inceptionv3(pretrained=self.args.fmpn_cn_pretrained, n_classes=self.args.n_classes)

        elif self.args.fmpn_cn == "scnn":
            #
            # laod an small cnn that will be used to classify the predicted masks
            #
            # do not forgett the following args
            #
            # args.fmpn_cn
            # args.scnn_config
            # args.n_classes
            # args.scnn_in_channels  ADD
            # args.scnn_llfeatures
            #
            # ADD ARGUMENT -> args.cls_masks
            #
            model = SCNN0(self.args)
            print(model)
            # print(summary(model, (1, self.args.final_size, self.args.final_size)))
            return SCNN0(self.args)

    def loss_total_fn(self, loss_fmg, loss_cn):
        """0.1, 1 are lambda1, lambda2 - add to args!!!"""
        return self.args.lambda_fmg * loss_fmg + self.args.lambda_cn * loss_cn  # lambda1, lambda2 to args!

    def __adjust_lr__(self):
        """Adjust the learning rate.
        1) Training starts by tuning only FMG for 300 epochs using adam. The lr is decayed linearly to 0 from epoch 150.
        2) After that train the whole fmpn for 200 epochs jointly and linearly decay the lr from epoch 100.
        But first reset learning rate of fmg to 0.00001 while rest uses 0.0001. Step 1) is done by first run of the fmg
        trained by a single run."""
        if self.args.scheduler_type == "linear_x":
            factor_cn = (self.args.lr_init - self.args.lr_end) / (self.args.epochs - self.args.start_lr_drop_fmpn)
            factor_fmg = (self.args.lr_init_after - self.args.lr_end) / (
                    self.args.epochs - self.args.start_lr_drop_fmpn)

            # print("adjusting factor cn: ", factor_cn)
            # print("adjusting factor fmg: ", factor_fmg)

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
        elif self.args.scheduler_type == "const":
            self.opt.param_groups[0]['lr'] = self.args.lr_init_after
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
            opt = torch.optim.Adam(self.fmg.parameters(), lr=self.args.lr_init, betas=(self.args.beta1, 0.9999))
        return opt

    def __set_device__(self):
        print("Set GPU device...")
        # device = torch.device(self.args.gpu_id)
        device = torch.device('cuda:%d' % self.args.gpu_id if torch.cuda.is_available() else 'cpu')
        print("GPU device: ", device)
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
            print("Loaded ckpt of pretrained fmg.")
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
        super(FmpnAgentMod, self).__create_folders__(self.name)

    def save_resultlists_as_dict(self, path):
        print("Saving current results as dict...\n")
        # print("train_loss: \n", self.list_train_loss)
        # print("train acc: \n", self.list_train_acc)
        # print("val loss: \n", self.list_test_loss)
        # print("val acc: \n", self.list_test_acc)
        dict = {
            'train_loss': self.list_train_loss,
            'train_loss_fmg': self.list_train_loss_fmg,
            'train_loss_cn': self.list_train_loss_cn,
            'train_acc': self.list_train_acc,
            'test_acc': self.list_test_acc,
            'test_loss': self.list_test_loss,
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
        # self.test()

    def train(self):
        print("start training loop...")
        with trange(self.tmp_epoch, self.args.epochs, desc="Epoch", unit="epoch") as epochs:
            for epoch in epochs:
                # print("epoch nr: ", epoch)
                self.fmg.train()
                # self.pfn.train()
                self.cn.train()

                self.tmp_epoch = epoch
                self.epoch_counter += 1

                # train one epoch
                epoch_loss, epoch_acc, epoch_loss_fmg, epoch_loss_cn = self.train_epoch()
                epoch_total_val_loss, epoch_val_acc, epoch_fmg_val_loss, epoch_cn_val_loss = self.eval_epoch()

                # append training learning rates for fmg and cn and total
                self.list_train_loss.append(epoch_loss)
                self.list_train_loss_fmg.append(epoch_loss_fmg)
                self.list_train_loss_cn.append(epoch_loss_cn)

                self.list_lr_fmg.append(self.opt.param_groups[0]['lr'])
                self.list_lr_pfn.append(self.opt.param_groups[1]['lr'])
                self.list_lr_cn.append(self.opt.param_groups[2]['lr'])

                self.list_train_acc.append(epoch_acc)

                # append testing results to arrays
                self.list_test_loss.append(epoch_total_val_loss)
                self.list_test_loss_fmg.append(epoch_fmg_val_loss)
                self.list_test_loss_cn.append(epoch_cn_val_loss)
                self.list_test_acc.append(epoch_val_acc)

                self.__adjust_lr__()

                # :TODO: set the max_acc so far if it is the case
                #   save this epoch if it is the best so far
                #   and the train accuracy is nearly 1.0
                if self.max_acc < epoch_val_acc:
                    self.max_acc = epoch_val_acc
                    if epoch_acc > 0.9899:
                        # save best tmp checkpoint
                        print("Saving best checkpoint so far...")
                        self.save_ckpt()
                        self.save_resultlists_as_dict(
                            self.train_path + "/" + "epoch_" + str(epoch) + "_train_logs.pickle")

                epochs.set_postfix(loss="{:.3f}".format(epoch_loss, prec='.3'),
                                   accuracy="{:.3f}".format(epoch_acc.item(), prec='.3'),
                                   val_loss="{:.3f}".format(epoch_total_val_loss, prec='.3'),
                                   val_accuracy="{:.3f}".format(epoch_val_acc, prec='.3'),
                                   val_acc_best="{:.3f}".format(self.max_acc, prec='.3'),
                                   lr_fmg="{:.6f}".format(self.list_lr_fmg[-1], prec='.6'),
                                   lr_cn="{:.6f}".format(self.list_lr_cn[-1], prec='.6'))

                if epoch % self.args.save_ckpt_intv == 0:
                    self.save_ckpt()
                    self.save_resultlists_as_dict(self.train_path + "/" + "epoch_" + str(epoch) + "_train_logs.pickle")

            # save last ckpt
            self.save_ckpt()
            self.train_logs_path = self.train_path + "end_train_logs.pickle"
            self.save_resultlists_as_dict(self.train_logs_path)

    def train_epoch(self):
        epoch_loss = 0.0
        epoch_fmg_loss = 0.0
        epoch_cn_loss = 0.0
        epoch_acc = 0.0

        for i, batch in enumerate(self.train_dl):
            images = batch["image"].to(self.device)
            images_gray = batch["image_gray"].to(self.device)
            label_masks = batch["mask"].to(self.device)
            labels = batch["label"].to(self.device)
            # paths = batch["img_path"]

            self.opt.zero_grad()

            predicted_masks = self.fmg(images_gray).to(self.device)

            # 1. Experiment
            # CONCATENATE ORIGINAL IMAGE AND PREDICTED MASK DEPTH WISE n_channels = 4
            # 2. Experiment
            # Provide just the mask to a simple classifier!
            #
            # dim=1 because pytorch tensor img have shape (n_imgs, n_channels, h, w)

            if self.args.cls_masks:  # just provide masks to classifier
                heat_face = predicted_masks
            else:  # provide image concatenated depthwise to classifier. Not by multiplication.
                heat_face = torch.cat((images, predicted_masks), dim=1)

            heat_face.to(self.device)
            # fusion_img, imgorg_after_pfn_prep, imgheat_after_pfn_prep = self.pfn(images, heat_face)

            # save images
            if self.args.save_samples:
                # :TODO: make an array with epochs when to save images e.g. [299, 350, 400, 450, 499]
                # if self.tmp_epoch == 299 or self.tmp_epoch % 100 == 0:
                if self.tmp_epoch in [299, 349, 399, 449, 498]:
                # if self.tmp_epoch in [0, 49, 99, 149, 199]:
                    if i < 2:  # nr_batches to save
                        print("start saving samples")
                        for idx in range(len(predicted_masks)):
                            save_tensor_img(img=images[idx].cpu().detach(),
                                            path=self.train_plots + "imgorg_"
                                                 + str(labels[idx].cpu().detach().numpy()) + "_epoch_" + str(
                                                self.tmp_epoch) + "_batch_" + str(i) + ".png")

                            save_tensor_img(img=predicted_masks[idx].cpu().detach(),
                                            path=self.train_plots + "pred_mask_"
                                                 + str(labels[idx].cpu().detach().numpy()) + "_epoch_" + str(
                                                self.tmp_epoch) + "_batch_" + str(i) + ".png")

                            save_tensor_img(img=images_gray[idx].cpu().detach(),
                                            path=self.train_plots + "gray_img_"
                                                 + str(labels[idx].cpu().detach().numpy()) + "_epoch_" + str(
                                                self.tmp_epoch) + "_batch_" + str(i) + ".png")

                            #save_tensor_img(img=heat_face[idx].cpu().detach(),
                            #                path=self.train_plots + "multiplied_img_"
                            #                     + str(labels[idx].cpu().detach().numpy()) + "_epoch_" + str(
                            #                    self.tmp_epoch) + "_batch_" + str(i) + ".png")

            classifications = self.cn(heat_face)

            # apply softmax to logits
            if self.args.fmpn_cn == "inc_v3":
                classifications_soft = torch.softmax(classifications.logits, dim=-1)
                cn_loss = self.loss_cn_fn(classifications.logits, labels)
            else:
                classifications_soft = torch.softmax(classifications, dim=-1)
                cn_loss = self.loss_cn_fn(classifications, labels)

            fmg_loss = self.loss_fmg_fn(predicted_masks, label_masks)
            # cn_loss = self.loss_cn_fn(classifications.logits, labels)

            epoch_fmg_loss += fmg_loss.item()
            epoch_cn_loss += cn_loss.item()

            total_loss = self.loss_total_fn(fmg_loss, cn_loss)

            total_loss.backward()
            self.opt.step()

            epoch_loss += total_loss.item()
            epoch_acc += self.calc_accuracy(classifications_soft, labels)
        epoch_acc = epoch_acc / len(self.train_dl)

        epoch_fmg_loss = epoch_fmg_loss / len(self.train_dl)
        epoch_cn_loss = epoch_cn_loss / len(self.train_dl)

        epoch_loss = epoch_loss / len(self.train_dl)
        return epoch_loss, epoch_acc, epoch_fmg_loss, epoch_cn_loss

    def eval_epoch(self):
        self.fmg.eval()
        self.pfn.eval()
        self.cn.eval()

        epoch_fmg_val_loss = 0.0
        epoch_cn_val_loss = 0.0
        epoch_total_val_loss = 0.0
        epoch_val_acc = 0.0

        with torch.no_grad():
            for i, batch in enumerate(self.test_dl):  #
                images = batch["image"].to(self.device)
                images_gray = batch["image_gray"].to(self.device)
                label_masks = batch["mask"].to(self.device)
                labels = batch["label"].to(self.device)

                predicted_masks = self.fmg(images_gray).to(self.device)

                if self.args.cls_masks:
                    heat_face = predicted_masks
                else:
                    heat_face = torch.cat((images, predicted_masks), dim=1)

                heat_face.to(self.device)

                # fusion_img, _, __ = self.pfn(images, heat_face)

                classifications = self.cn(heat_face)
                # print("classifications: ", classifications)
                # print("classification shape:", np.shape(classifications))
                classification_prob = torch.softmax(classifications, dim=-1)
                # print("class after softmax: ", classification_prob)

                #
                # classifications = classification_prob
                #
                fmg_loss = self.loss_fmg_fn(predicted_masks, label_masks)

                # classifications_argmax = torch.argmax(classifications)
                # print("classes: ", classifications)
                cn_loss = self.loss_cn_fn(classifications, labels)

                epoch_fmg_val_loss += fmg_loss.item()
                epoch_cn_val_loss += cn_loss.item()

                total_loss = self.loss_total_fn(fmg_loss, cn_loss)
                epoch_val_acc += self.calc_accuracy(classification_prob, labels)

                epoch_total_val_loss += total_loss.item()

            epoch_fmg_val_loss = epoch_fmg_val_loss / len(self.test_dl)
            epoch_cn_val_loss = epoch_cn_val_loss / len(self.test_dl)
            epoch_total_val_loss = epoch_total_val_loss / len(self.test_dl)
            epoch_val_acc = epoch_val_acc / len(self.test_dl)

        return epoch_total_val_loss, epoch_val_acc, epoch_fmg_val_loss, epoch_cn_val_loss

    def calc_accuracy(self, predictions, labels):
        """take softmax(logits) outputs"""
        classes = torch.argmax(predictions, dim=-1)
        # print("Classes after argmax: \n", classes)
        return torch.mean((classes == labels).float())

    def inference(self, batch):
        self.fmg.eval()
        self.pfn.eval()
        self.cn.eval()

        images = batch["image"].to(self.device)
        images_gray = batch["image_gray"].to(self.device)
        label_masks = batch["mask"].to(self.device)
        labels = batch["label"].to(self.device)

        predicted_masks = self.fmg(images_gray).to(self.device)
        heat_face = torch.cat((images, predicted_masks), dim=1)

        heat_face.to(self.device)
        # fusion_img, a, b = self.pfn(images, heat_face)

        classifications = self.cn(heat_face)
        # print("classifications: ", classifications)
        # print("classification shape:", np.shape(classifications))
        classification_prob = torch.softmax(classifications, dim=-1)

        return classification_prob, labels, heat_face

    def test(self):
        print("testing...")
        self.fmg.eval()
        self.pfn.eval()
        self.cn.eval()

        epoch_fmg_val_loss = 0.0
        epoch_cn_val_loss = 0.0
        epoch_total_val_loss = 0.0
        epoch_val_acc = 0.0

        #  for confusion matrix
        all_predictions = torch.tensor([]).to(self.device)
        all_labels = torch.tensor([]).to(self.device)

        with torch.no_grad():
            for i, batch in enumerate(self.test_dl):  #
                images = batch["image"].to(self.device)
                images_gray = batch["image_gray"].to(self.device)
                label_masks = batch["mask"].to(self.device)
                labels = batch["label"].to(self.device)

                predicted_masks = self.fmg(images_gray).to(self.device)

                if self.args.cls_masks:
                    heat_face = predicted_masks
                else:
                    heat_face = torch.cat((images, predicted_masks), dim=1)

                heat_face.to(self.device)
                # fusion_img, a, b = self.pfn(images, heat_face)
                classifications = self.cn(heat_face)
                # print("classifications: ", classifications)
                # print("classification shape:", np.shape(classifications))
                classification_prob = torch.softmax(classifications, dim=-1)
                # print("class after softmax: ", classification_prob)

                #
                # classifications = classification_prob
                #

                fmg_loss = self.loss_fmg_fn(predicted_masks, label_masks)

                # classifications_argmax = torch.argmax(classifications)
                # print("classes: ", classifications)
                cn_loss = self.loss_cn_fn(classifications, labels)

                epoch_fmg_val_loss += fmg_loss.item()
                epoch_cn_val_loss += cn_loss.item()

                total_loss = self.loss_total_fn(fmg_loss, cn_loss)

                epoch_total_val_loss += total_loss.item()

                # pass classifications to all_predictions tensor
                # all_predictions = torch.cat((all_predictions, torch.argmax(classifications, dim=-1)))
                all_predictions = torch.cat((all_predictions, torch.argmax(classification_prob, dim=-1)))
                all_labels = torch.cat((all_labels, labels))

                batch_val_acc = self.calc_accuracy(classification_prob, labels)
                epoch_val_acc += batch_val_acc

                # tbatch.set_postfix(val_loss="{:.3f}".format(total_loss, prec='.3'),
                #                    val_accuracy="{:.3f}".format(batch_val_acc, prec='.3'))

            epoch_total_val_loss = epoch_total_val_loss / len(self.test_dl)
            epoch_val_acc = epoch_val_acc / len(self.test_dl)
            # print("val_acc: ", epoch_val_acc)
            # print("val_loss: {0} \t val_accuracy: {1}".format(epoch_total_val_loss, epoch_val_acc))

            #
            # NOTE: FMPN does not support neutral class! Therefore no case for ==7
            #
            if self.args.dataset == "ckp":
                classnames = None
            elif self.args.dataset == "fer":  # 6 classes without neutral image
                if self.args.n_classes == 6:
                    classnames = ["anger", "disgust", "fear", "happy", "sadness", "surprise"]
            elif self.args.dataset == "rafdb":
                classnames = ['surprise', 'fear', 'disgust', 'happiness', 'sadness', 'anger']

            """
             0 surprise, 1 fear, 2 disgust, 3 happiness, 4 sadness, 5 anger, 6 neutral
            """

            cnfmat = make_cnfmat_plot(labels=all_labels,
                                      predictions=all_predictions,
                                      n_classes=self.args.n_classes,
                                      path=self.test_plots,
                                      gpu_device=self.args.gpu_id,
                                      dataset=self.args.dataset,
                                      classnames=classnames)

            # sum diagonal of cnf matrix and sum whole matrix
            # then divide correctly predicted by all predictions
            diag = np.trace(cnfmat)
            all_preds = np.sum(cnfmat)
            acc = diag / all_preds
            #
            # :TODO: calculate metrics here
            #
            # calculate precision recall fscore
            clf_report = prec_recall_fscore(y_true=all_labels.cpu(), y_pred=all_predictions.cpu())
            roc_score = roc_auc_score(y_true=all_labels.cpu(), y_pred=all_predictions.cpu(),
                                      n_classes=self.args.n_classes)
            # out_df.to_csv(self.test_plots + "clf_report.csv", index=False)
            out_dict = {
                "precision": clf_report[0].round(2).tolist(),
                "recall": clf_report[1].round(2).tolist(),
                "f1": clf_report[2].round(2).tolist(),
                "support": clf_report[3].tolist(),
                "roc_auc_ovr": roc_score.tolist(),
                "test_acc": acc
            }
            with open(self.test_plots + "clf_report.txt", "w") as f:
                print(out_dict, file=f)
            print(out_dict)
            print(roc_score)
            return epoch_total_val_loss, epoch_val_acc

            # creating confusion matrix now
            #   print(all_predictions)
            #   print(all_labels)
            #   print("allpred: ", np.shape(all_predictions))
            #   print("allpred: ", np.shape(all_labels))
            #   stacked_pred_label = torch.stack((all_labels, all_predictions), dim=1)
            #   print(stacked_pred_label.shape)
            #   print(stacked_pred_label)
            #   for pair in stacked_pred_label:
            #       label, pred = pair.tolist()
            #       cnfmat[int(label), int(pred)] = cnfmat[int(label), int(pred)] + 1

            #   print(cnfmat)
            # save confusion matrix to file
            # path = "./results/run_fmpn_2021-04-16_00-40-16/train_fmpn_2021-04-16_00-40-16\plots/"
            #   path = "./results/run_fmpn_2021-04-19_19-10-27/train_fmpn_2021-04-19_19-10-27\plots/"
            #   np.savetxt(path + "cnfmat.txt", cnfmat.numpy())
            #   cnfmat = cnfmat.detach().cpu().numpy()
            #   cnfmatnorm = cnfmat.astype('float') / cnfmat.sum(axis=1)[:, np.newaxis]
            #   cnfmat_df = pd.DataFrame(cnfmatnorm,
            #                            index=["anger", "contempt", "disgust", "fear", "happy", "sadness", "surprise"],
            #                            columns=["anger", "contempt", "disgust", "fear", "happy", "sadness",
            #                                     "surprise"])

            #   ax = sn.heatmap(cnfmat_df, annot=True, annot_kws={"size": 10}, fmt='.0%', cmap='Blues')
            #   plt.title('Confusion matrix of FMPN predictions on the CK+ dataset', fontsize=12)
            #   plt.yticks(rotation=0)
            #   plt.savefig(path + "cnfmat_plot.png")
            #   plt.show()
