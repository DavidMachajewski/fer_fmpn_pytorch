# from args import Setup
from args2 import Setup

import tqdm
from tqdm import tqdm
import os
from lib.dataloader.datasets import CKP, RandomCrop, ToTensor, GrayScale
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms, utils
import torchvision
# from lib.models.models import BuildingBlock, ResNet18, FMPN

from torchviz import make_dot
from torch.nn import Sequential
from torch import nn
from torchsummary import summary
import torch.optim as optim
import torch as to
from lib.dataloader.datasets import get_ckp

def resnet_train(model, dataloader):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    device = to.device("cuda:0" if to.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(3):

        running_loss = 0.0
        for i, data in enumerate(dataloader, 0):
            images, labels = data["image"], data["label"]
            images.to(device)
            labels.to(device)
            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 10 == 0:  # print every 2000 mini-batches
                print("predictions mat: \n", outputs)
                print("predictions: \n", outputs.max(1).indices)
                print("labels: \n", labels)
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2))
                running_loss = 0.0

def fmg_train(model, dataloader):
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    device = to.device("cuda:0" if to.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(3):

        running_loss = 0.0
        for i, data in enumerate(dataloader, 0):
            images, labels = data["image_gray"], data["mask"]
            images.to(device)
            labels.to(device)
            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2 == 0:  # print every 2000 mini-batches
                print("predictions mat: \n", outputs)
                print("predictions: \n", outputs.max(1).indices)
                print("labels: \n", labels)
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2))
                running_loss = 0.0


def run_fmg_agent():
    args = Setup().parse()
    args.epochs = 300
    args.start_lr_drop = 150
    args.model_to_train = "fmg"
    args.batch_size = 8
    args.ckpt_to_load = "./results/run_fmg_2021-04-14_13-57-45/train_fmg_2021-04-14_13-57-45\ckpt/fmg_2021-04-14_13-57-45_epoch_299_ckpt.pth.tar"
    args.load_ckpt = False

    fmgagent = FmgAgent(args)
    # fmgagent.run()
    fmgagent.test("./results/run_fmg_2021-04-14_13-57-45/test_fmg_2021-04-14_13-57-45\plots/")


def run_fmpn_agent():
    args = Setup().parse()
    args.epochs = 500
    args.start_lr_drop = 400
    args.batch_size = 8

    # args.load_ckpt_fmg_only = False
    # args.ckpt_fmg = "./results/run_fmg_2021-04-14_13-57-45/train_fmg_2021-04-14_13-57-45\ckpt/fmg_ckpt.pth.tar"

    args.load_ckpt = True

    args.ckpt_fmg = "./results/run_fmpn_2021-04-15_17-52-38/train_fmpn_2021-04-15_17-52-38\ckpt/fmpn_fmg_ckpt.pth.tar"
    args.ckpt_pfn = "./results/run_fmpn_2021-04-15_17-52-38/train_fmpn_2021-04-15_17-52-38\ckpt/fmpn_pfn_ckpt.pth.tar"
    args.ckpt_cn = "./results/run_fmpn_2021-04-15_17-52-38/train_fmpn_2021-04-15_17-52-38\ckpt/fmpn_cn_ckpt.pth.tar"

    fmpn_agent = FmpnAgent(args)
    fmpn_agent.run()


from lib.utils import imshow_tensor
# from lib.models.models import RunFMPN
from lib.models.models2 import train_inceptionv3
from lib.agents.fmg_agent import FmgAgent
from lib.agents.fmpn_agent import FmpnAgent

if __name__ == '__main__':

    run_fmpn_agent()


"""
    train, test = get_ckp(args, batch_size=16)

    model = FMPN(args)
    model.setup()
    epochs_total = 1
    lr_per_epoch = []
    loss_fmg_per_epoch = []
    loss_cn_per_epoch = []
    for epoch in range(epochs_total):
        for i, batch in enumerate(tqdm(train)):
            model.feed(batch)
            model.optimizer_step()
            tmp_loss = model.get_latest_losses()
            print("loss: ", tmp_loss)
            cur_lr = model.update_learning_rate()
            info_dict = {
                'epoch': epoch,
                'epochs_total': epochs_total,
                'cur_lr': cur_lr,
                'losses': tmp_loss
            }
            # print(info_dict)
            loss_fmg_per_epoch.append(tmp_loss['fmg'])
            loss_cn_per_epoch.append(tmp_loss['cn'])
            lr_per_epoch.append(cur_lr)
            model.save(epoch)
            print(loss_fmg_per_epoch)
            print(loss_cn_per_epoch)
            print(lr_per_epoch)
            break

"""
