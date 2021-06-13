"""
####################################################################################################
train Inception_v3 network on ck+
training is done in 10 split so change trainsplit
and testsplit accordingly: train_ids_x.csv, test_ids_x.csv
####################################################################################################
python main.py --mode train --gpu_id 0 --model_to_train incv3 --epochs 200 --pretrained False
--dataset ckp --batch_size 8 --lr_gen 0.001 --trainsplit train_ids_2.csv --testsplit test_ids_2.csv

to run multiple times with other values you can run following bash command:
for %i in (3,4,5) do python main.py --mode train --gpu_id 0 --model_to_train incv3 --epochs 200 --pretrained False --dataset ckp --batch_size 8 --lr_gen 0.001 --trainsplit train_ids_%i.csv --testsplit test_ids_%i.csv
####################################################################################################
"""
from args2 import Setup
from torch import nn
import torch.optim as optim
import torch as to

from lib.agents.fmg_agent import FmgAgent

"""
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
"""

"""
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
"""


def run_fmg_agent():
    args = Setup().parse()
    args.epochs = 300  # 300
    args.start_lr_drop = 150
    args.model_to_train = "fmg"
    args.batch_size = 8
    args.trainsplit = "train_ids_0.csv"
    args.testsplit = "test_ids_0.csv"
    # args.ckpt_to_load = "./results/run_fmg_2021-04-14_13-57-45/train_fmg_2021-04-14_13-57-45\ckpt/fmg_2021-04-14_13-57-45_epoch_299_ckpt.pth.tar"
    args.ckpt_to_load = "./results/run_fmg_2021-05-20_10-45-40/train_fmg_2021-05-20_10-45-40/ckpt/fmg_2021-05-20_10-45-40_epoch_120_ckpt.pth.tar"
    # ./results/run_fmg_2021-05-20_10-45-40/train_fmg_2021-05-20_10-45-40/ckpt/fmg_2021-05-20_10-45-40_epoch_120_ckpt.pth.tar
    args.load_ckpt = 0

    fmgagent = FmgAgent(args)
    fmgagent.run()
    #fmgagent.test("./results/run_fmg_2021-04-14_13-57-45/test_fmg_2021-04-14_13-57-45\plots/")



def run_fmpn_agent():
    args = Setup().parse()
    args.epochs = 500
    args.start_lr_drop = 400
    args.batch_size = 8

    # first training
    args.load_ckpt_fmg_only = 1
    args.ckpt_fmg = "./results/run_fmg_2021-04-14_13-57-45/train_fmg_2021-04-14_13-57-45\ckpt/fmg_ckpt.pth.tar"

    # for resuming training
    # args.load_ckpt = 1

    # args.ckpt_fmg = "./results/run_fmpn_2021-04-15_17-52-38/train_fmpn_2021-04-15_17-52-38\ckpt/fmpn_fmg_ckpt.pth.tar"
    # args.ckpt_pfn = "./results/run_fmpn_2021-04-15_17-52-38/train_fmpn_2021-04-15_17-52-38\ckpt/fmpn_pfn_ckpt.pth.tar"
    # args.ckpt_cn = "./results/run_fmpn_2021-04-15_17-52-38/train_fmpn_2021-04-15_17-52-38\ckpt/fmpn_cn_ckpt.pth.tar"

    # for testing ----
    # args.ckpt_fmg = "./results/run_fmpn_2021-04-16_00-40-16/train_fmpn_2021-04-16_00-40-16/fmpn_fmg_ckpt.pth.tar"
    # args.ckpt_pfn = "./results/run_fmpn_2021-04-16_00-40-16/train_fmpn_2021-04-16_00-40-16/fmpn_pfn_ckpt.pth.tar"
    # args.ckpt_cn = "./results/run_fmpn_2021-04-16_00-40-16/train_fmpn_2021-04-16_00-40-16/fmpn_cn_ckpt.pth.tar"

    # for testing ----
    args.ckpt_fmg = "./results/run_fmpn_2021-04-19_19-10-27/train_fmpn_2021-04-19_19-10-27\ckpt/fmpn_fmg_2021-04-19_19-10-27_epoch_499_ckpt.pth.tar"
    args.ckpt_pfn = "./results/run_fmpn_2021-04-19_19-10-27/train_fmpn_2021-04-19_19-10-27\ckpt/fmpn_pfn_2021-04-19_19-10-27_epoch_499_ckpt.pth.tar"
    args.ckpt_cn = "./results/run_fmpn_2021-04-19_19-10-27/train_fmpn_2021-04-19_19-10-27\ckpt/fmpn_cn_2021-04-19_19-10-27_epoch_499_ckpt.pth.tar"

    fmpn_agent = FmpnAgent(args)
    # fmpn_agent.run()
    fmpn_agent.test()


"""
def run_inc_agent():
    args = Setup().parse()
    args.model_to_train = "incv3"
    args.epochs = 200
    args.batch_size = 1

    args.load_ckpt = True
    args.ckpt_to_load = "./results/run_incv3_2021-04-21_19-29-37/train_incv3_2021-04-21_19-29-37\ckpt\incv3_epoch_199_ckpt.pth.tar"

    agent = InceptionAgent(args)
    # agent.run()
    res = agent.test()
    print(res)
"""


from lib.agents.runner import Runner
import matplotlib.pyplot as plt
from lib.featurevisualization.deepdream import DeepDream
import numpy as np
import PIL
from lib.dataloader.datasets import AffectNetSubset
from lib.dataloader.datasets import get_fer2013, get_affectnet


def ferexample(args):
    train_dl, test_dl = get_fer2013(args)
    for batch in train_dl:
        img = batch['image']
        label = batch['label']
        print("label: ", label)
        print("image: ", img)


if __name__ == '__main__':
    args = Setup().parse()
    """--- train networks ---"""
    # comment "train mask generator" section
    runner = Runner(args)
    runner.start()

    """--- train mask generator ---"""
    # comment "train networks" section
    # run_fmg_agent()

    #for image, label in next(iter(train_dl)):
    #    plt.imshow(image.permute(1,2,0))
    #plt.show()

"""
    # --- run DeepDream algorithm ---
    
    # load a model -> model needs a inference function
    # init deep dream class
    # python main.py --deepdream_model "incv3" --pretrained 1 --load_ckpt 1 --ckpt_to_load "F:\trainings2\inceptionnet\pretrained\8\run_incv3_2021-05-10_19-26-32\train_incv3_2021-05-
    # 10_19-26-32\ckpt\incv3_epoch_199_ckpt.pth.tar" --dataset ckp --batch_size 1
    dream = DeepDream(args)
    batch = next(iter(dream.train_dl))

    img = batch["image"].to(to.device('cuda:0'))
    img.requires_grad = True
    # print(img.size())
    # label = batch["label"]
    # print(img)
    test_img = img.clone()
    plt.imshow(test_img[0].detach().cpu().squeeze().permute(1,2,0))
    # plt.show()

    # print(f"Label: {label}")
    # output = dream.model.Conv2d_1a_3x3.conv(img)
    # print(output)
    merge = [test_img[0].clone().detach().cpu().squeeze().permute(1,2,0).numpy()]

    # dream.gradient_step(img)
    # dream.dream(img)

    # make for loop with the same batch
    # for more layer
    for i in range(5):
        imgs_array = dream.start_dreaming(img, layer_no=i)
        # fig = plt.figure(figsize=(10, 10))
        # plt.imshow(imgs_array[0])
        print(np.array((imgs_array[0])))
        merge.append(np.array((imgs_array[0]))/255)
        # plt.show()
    merged_img = np.hstack(tuple(merge))
    fig = plt.figure(figsize=(15, 15))
    plt.imshow(merged_img)
    plt.axis('off')
    # plt.show()
    plt.savefig("C:/root/uni/bachelor/inceptionnet_feature_extraction23.png", bbox_inches='tight')
"""









"""
python main.py --deepdream_model "incv3" --pretrained 1 --load_ckpt 1 --ckpt_to_load "F:\trainings2\inceptionnet\pretrained\8\run_incv3_2021-05-10_19-26-32\train_incv3_2021-05-
10_19-26-32\ckpt\incv3_epoch_199_ckpt.pth.tar" --batch_size 2

"""


"""
for %i in (0) do python main.py --mode train --gpu_id 0 --model_to_train densenet --epochs 200 --save_ckpt_intv 50 --load_size 245 --final_size 224 --pretrained 0 --dataset ckp --batch_size 8 --lr_gen 0.001 --trainsplit train_ids_%i.csv --testsplit test_ids_%i.csv
"""


    # train_dl, test_dl = get_ckp(args)
    #
    # :TODO: MAKE UTILS FUNCTION
    #    PLOT BARPLOT OF TRAIN AND TEST DATA
    #

    # classes = [0, 0, 0, 0, 0, 0, 0]
    # for i, batch in enumerate(test_dl):
    #     labels = batch['label']
    #     for label in labels:
    #         classes[int(label)] += 1
    # print(classes)
    # plt.bar([0, 1, 2, 3, 4, 5, 6], classes)
    # plt.show()

#
# run evaluation:
#
# python main.py --mode test --gpu_id 0 --model_to_train densenet --load_ckpt 1 --ckpt_to_load "./results/run_densenet_2021-05-11_15-36-08/train_densenet_2021-05-11_15-36-08/ckpt/densenet_epoch_199_ckpt.pth.tar" --load_size 245 --final_size 224 --dataset ckp --batch_size 8  --trainsplit train_ids_3.csv --testsplit test_ids_3.csv
#

#
#  RUN A TEST FOR EXAMPLE DENSENET ON TRAIN, TEST SPLIT 3
#  you have to load the last training checkpoint...
#
# python main.py --mode test --gpu_id 0 --model_to_train densenet --load_ckpt 1 --ckpt_to_load "./results/run_densenet_2021-05-11_15-36-08/train_densenet_2021-05-11_15-36-08/ckpt
# /densenet_epoch_199_ckpt.pth.tar" --load_size 245 --final_size 224 --dataset ckp --batch_size 8  --trainsplit train_ids_3.csv --testsplit test_ids_3.csv
