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

"""
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
"""


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



if __name__ == '__main__':
    args = Setup().parse()

    runner = Runner(args)

    runner.start()
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