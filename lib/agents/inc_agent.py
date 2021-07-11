import pickle

import numpy as np
import torch
import pandas as pd
from tqdm import tqdm, trange
from lib.models.models import get_ckp
from lib.dataloader.datasets import get_fer2013
from lib.agents.agent import Agent
from lib.models.models import inceptionv3
from lib.eval.eval_utils import make_cnfmat_plot, prec_recall_fscore, roc_auc_score


#
# :TODO:
# cross database experiments
#  https://arxiv.org/pdf/1511.04110.pdf


class InceptionAgent(Agent):
    def __init__(self, args):
        super(InceptionAgent, self).__init__(args)
        self.name = "incv3"
        self.args = args  # inception_pretrained to args
        self.model = inceptionv3(pretrained=self.args.pretrained)

        self.epoch_counter = 0

        if self.args.dataset == "ckp":
            self.train_dl, self.test_dl, self.valid_dl = get_ckp(args=self.args,
                                                                 batch_size=self.args.batch_size,
                                                                 shuffle=True,
                                                                 num_workers=self.args.num_workers,
                                                                 drop_last=True,
                                                                 valid=True)
            print("Loaded ckp dataset.")
        elif self.args.dataset == "fer":
            self.train_dl, self.test_dl = get_fer2013(args=self.args,
                                                      batch_size=self.args.batch_size,
                                                      shuffle=True,
                                                      num_workers=self.args.num_workers,
                                                      drop_last=True)
            print("Loaded fer dataset.")

        self.opt = self.__init_optimizer__()
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.device = self.__set_device__()
        self.model.to(self.device)
        self.train_logs_path = None

        if self.args.load_ckpt:
            self.load_ckpt(self.args.ckpt_to_load)

    def __init_optimizer__(self):
        if self.args.optimizer == "adam":
            opt = torch.optim.Adam(self.model.parameters(),
                                   # lr=self.args.lr_init,
                                   lr=self.args.lr_gen,
                                   betas=(self.args.beta1, 0.9999))
        return opt

    def __set_device__(self):
        if self.is_cuda:
            device = torch.device(self.args.gpu_id)
        else:
            device = torch.device("cpu")
        return device

    def load_ckpt(self, file_name):
        try:
            ckpt = torch.load(file_name)
            self.tmp_epoch = ckpt['epoch']
            self.model.load_state_dict(ckpt['model_state_dict'])
            self.opt.param_groups[0] = ckpt['model_optimizer']
            print("Loaded checkponint {}".format(file_name))
        except OSError as e:
            print("Could not load load checkpoint.")

    def save_ckpt(self):
        file_name = self.name + "_epoch_" + str(self.tmp_epoch) + "_ckpt.pth.tar"

        state = {
            'epoch': self.tmp_epoch,
            'model_state_dict': self.model.state_dict(),
            'model_optimizer': self.opt.param_groups[0]
        }
        torch.save(state, self.train_ckpt + file_name)

    def __create_folders__(self):
        super(InceptionAgent, self).__create_folders__(self.name)

    def save_resultlists_as_dict(self, path):
        print("\nSaving current results as dict...\n")
        # print("train_loss: \n", self.list_train_loss)
        # print("train_acc: \n", self.list_train_acc)
        # print("val_loss: \n", self.list_test_loss)
        # print("val_acc: \n", self.list_test_acc)
        dict = {
            'train_loss': self.list_train_loss,
            'train_acc': self.list_train_acc,
            'test_loss': self.list_test_loss,
            'test_acc': self.list_test_acc,
            'lr': self.list_lr,
        }
        file = open(path, "wb")
        pickle.dump(dict, file)

    def train(self):
        with trange(self.tmp_epoch, self.args.epochs, desc="Epoch", unit="epoch") as epochs:
            for epoch in epochs:
                self.model.train()

                self.tmp_epoch = epoch
                self.epoch_counter += 1

                # train/test one epoch
                train_loss, train_acc = self.train_epoch()
                test_loss, test_acc = self.eval_epoch()

                # append to lists
                self.list_train_loss.append(train_loss)
                self.list_train_acc.append(train_acc)
                self.list_test_loss.append(test_loss)
                self.list_test_acc.append(test_acc)
                self.list_lr.append(self.opt.param_groups[0]['lr'])

                self.__adjust_lr__()

                if self.max_acc < test_acc:
                    self.max_acc = test_acc
                    if test_acc > 0.9899:
                        print("Saving best checkpoint so far...")
                        self.save_ckpt()
                        self.save_resultlists_as_dict(
                            self.train_path + "/" + "epoch_" + str(epoch) + "_train_logs.pickle")

                epochs.set_postfix(loss="{:.3f}".format(train_loss, prec='.3'),
                                   acc="{:.3f}".format(train_acc, prec='.3'),
                                   val_loss="{:.3f}".format(test_loss, prec='.3'),
                                   val_acc="{:.3f}".format(test_acc, prec='.3'))

                if epoch % self.args.save_ckpt_intv == 0:
                    self.save_ckpt()
                    self.save_resultlists_as_dict(self.train_path + "/" + "epoch_" + str(epoch) + "_train_logs.pickle")
            # save last checkpoint on training end as well
            self.save_ckpt()
            self.train_logs_path = self.train_path + "end_train_logs.pickle"
            self.save_resultlists_as_dict(self.train_logs_path)

    def train_epoch(self):
        """Train loop for one epoch."""
        epoch_loss, epoch_acc = 0.0, 0.0

        for i, batch in enumerate(self.train_dl):
            # print("batch {0} out of {1}".format(i, len(self.train_dl)))
            images = batch["image"].to(self.device)
            # print("image: ", images)
            labels = batch["label"].to(self.device)
            # print("label: ", labels)

            self.opt.zero_grad()

            cls_soft = torch.softmax(
                self.model(images).logits,
                dim=-1
            )

            loss = self.loss_fn(cls_soft, labels)

            loss.backward()
            self.opt.step()

            epoch_loss += loss.item()
            epoch_acc += self.__calc_accuracy__(cls_soft, labels)
        epoch_loss = epoch_loss / len(self.train_dl)
        epoch_acc = epoch_acc / len(self.train_dl)
        return epoch_loss, epoch_acc

    def eval_epoch(self):
        """Test loop for one epoch. Used while network is training."""
        self.model.eval()

        epoch_val_loss, epoch_val_acc = 0.0, 0.0

        with torch.no_grad():
            for i, batch in enumerate(self.test_dl):
                images = batch["image"].to(self.device)
                labels = batch["label"].to(self.device)

                cls_soft = torch.softmax(
                    self.model(images),
                    dim=-1
                )

                loss = self.loss_fn(cls_soft, labels)

                epoch_val_loss += loss
                epoch_val_acc += self.__calc_accuracy__(cls_soft, labels)
            epoch_val_loss = epoch_val_loss / len(self.test_dl)
            epoch_val_acc = epoch_val_acc / len(self.test_dl)
        return epoch_val_loss, epoch_val_acc

    def test(self):
        """Independent test loop.
        :return:
        """
        self.model.eval()
        epoch_val_loss, epoch_val_acc = 0.0, 0.0

        #  for confusion matrix
        all_predictions = torch.tensor([]).to(self.device)
        all_labels = torch.tensor([]).to(self.device)

        with torch.no_grad():
            for i, batch in enumerate(self.test_dl):
                images = batch["image"].to(self.device)
                labels = batch["label"].to(self.device)

                cls_soft = torch.softmax(
                    self.model(images),
                    dim=-1
                )

                loss = self.loss_fn(cls_soft, labels)
                epoch_val_loss += loss
                epoch_val_acc += self.__calc_accuracy__(cls_soft, labels)

                # pass this to the cnf_mat() function
                all_predictions = torch.cat((all_predictions, torch.argmax(cls_soft, dim=-1)))
                all_labels = torch.cat((all_labels, labels))

            epoch_val_loss = epoch_val_loss / len(self.test_dl)
            epoch_val_acc = epoch_val_acc / len(self.test_dl)

            # create heatplot from confusion matrix
            cnfmat = make_cnfmat_plot(labels=all_labels,
                                      predictions=all_predictions,
                                      n_classes=self.args.n_classes,
                                      path=self.test_plots,
                                      gpu_device=self.args.gpu_id)

            diag = np.trace(cnfmat)
            all_preds = np.sum(cnfmat)
            acc = diag/all_preds  # test acc

            # calculate precision recall fscore
            clf_report = prec_recall_fscore(y_true=all_labels.cpu(), y_pred=all_predictions.cpu())
            roc_score = roc_auc_score(y_true=all_labels.cpu(), y_pred=all_predictions.cpu(),
                                      n_classes=self.args.n_classes)

            out_dict = {
                "precision": clf_report[0].round(2),
                "recall": clf_report[1].round(2),
                "f1": clf_report[2].round(2),
                "support": clf_report[3],
                "roc_auc_ovr": roc_score.tolist(),
                "test_acc": acc
            }
            # out_df = pd.DataFrame(out_dict, index=self.args.n_classes)
            # avg_tot = (
            #     out_df.apply(lambda x: round(x.mean(), 2) if x.name != "support" else round(x.sum(), 2)).to_frame().T)
            # avg_tot.index = ["avg/total"]
            # out_df = out_df.append(avg_tot)
            # save to file
            # out_df.to_csv(self.test_plots + "clf_report.csv", index=False)

            with open(self.test_plots + "clf_report.txt", "w") as f:
                print(out_dict, file=f)

            print(out_dict)
            print(roc_score)

            return epoch_val_loss, epoch_val_acc

    def run(self):
        self.__create_folders__()
        self.save_args(self.run_path + "args.txt")
        self.train()

    def __calc_accuracy__(self, predictions, labels):
        classes = torch.argmax(predictions, dim=-1)
        return torch.mean((classes == labels).float())

    def __adjust_lr__(self):
        pass
