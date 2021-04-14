from lib.dataloader.datasets import get_ckp
import torch
import lib.models.models as cm
import torchvision as tv
from args import Setup
import numpy as np
from tqdm import tqdm
from tqdm import trange


def train_inceptionv3(epochs):
    args = Setup().parse()

    # args.load_size = 250
    # args.final_size = 224

    model = tv.models.inception_v3(pretrained=False, init_weights=True)
    device = torch.device("cuda:0")
    model.to(device)

    train_loader, test_loader = get_ckp(args)

    criterion = torch.nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)

    model.train()

    with trange(0, epochs, desc="Epoch", unit="epoch") as epochs:
        for epoch in epochs:
            epochs.set_description(f"Epoch {epoch}")

            adjust_learning_rate(optimizer, epoch)
            correct = 0  # calculating accuracy
            running_acc = 0.0
            epoch_loss = 0.0
            running_loss = 0.0
            for i, batch in enumerate(train_loader):
                image = batch["image"].to(device)
                label = batch["label"].to(device)

                optimizer.zero_grad()

                output = model(image)

                loss = criterion(output.logits, label)

                # output = torch.argmax(output.logits, dim=-1)

                # prec1, prec5 = accuracy(output.logits, label, topk=(1, 5))

                loss.backward()
                optimizer.step()

                # correct += (output == label).float().sum()

                epoch_loss += loss.item()
                running_loss += loss.item()
                running_acc += calc_accuracy(output.logits, label)
            # print("len trainl: ", len(train_loader))
            running_acc = running_acc / len(train_loader)
            epoch_loss = epoch_loss/len(train_loader)
            # print("Training epoch {} - loss: {:.3f} \t accuracy: {:.3f}".format(epoch+1, epoch_loss/(len(label)*len(train_loader)), running_acc.item(), prec='.3'))
            epochs.set_postfix(loss="{:.3f}".format(epoch_loss, prec='.3'),
                               accuracy="{:.3f}".format(running_acc.item(), prec='.3'))


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def calc_accuracy(predictions, labels):
    classes = torch.argmax(predictions, dim=-1)
    return torch.mean((classes == labels).float())


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 1 epochs"""
    lr = 0.1 * (0.1 ** (epoch // 1))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train_vgg(epochs):
    tv.models.vgg16(pretrained=True)


def train_resnet(epochs):
    tv.models.resnet18(pretrained=True)