import torch
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from pickle import load
from sklearn import metrics
from torch.nn.functional import one_hot


def make_ds_distribution_plot(train_dl, test_dl, save_to, n_classes=7, valid_dl=None, classnames = None):
    fig, ax = plt.subplots()
    idx = np.arange(n_classes)

    # train count test count
    trc = [0] * n_classes
    tec = [0] * n_classes
    vac = [0] * n_classes

    for i, batch in enumerate(train_dl):
        labels = batch['label']
        for label in labels:
            trc[int(label)] += 1

    for i, batch in enumerate(test_dl):
        labels = batch['label']
        for label in labels:
            tec[int(label)] += 1

    if valid_dl is not None:
        for i, batch in enumerate(valid_dl):
            labels = batch['label']
            for label in labels:
                vac[int(label)] += 1

    train_bar = ax.bar(idx, trc, label='train', bottom=np.array(tec) + np.array(vac))
    test_bar = ax.bar(idx, tec, label='test', bottom=np.array(vac))
    if valid_dl is not None:
        valid_bar = ax.bar(idx, vac, label='valid')
    # train_bar = ax.bar(idx, trc, label='train')
    # test_bar = ax.bar(idx, tec, label='test')
    # valid_bar = ax.bar(idx, vac, label='valid')

    # ax.axhline(0, color='grey', linewidth=0.8)
    ax.set_ylabel('number of samples')
    ax.set_title('Size of train, test, valid split per class')
    ax.set_xticks(idx)
    if classnames is not None:
        classnames = tuple(classnames)
    else:
        classnames = ('anger', 'contempt', 'disgust', 'fear', 'happy', 'sadness', 'surprise')
    ax.set_xticklabels(classnames)

    ax.legend(frameon=False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.xaxis.set_ticks_position('none')

    ax.bar_label(test_bar, label_type='edge', fontsize=5)
    ax.bar_label(train_bar, label_type='edge', fontsize=5)
    if valid_dl is not None:
        ax.bar_label(valid_bar, label_type='edge', fontsize=5)

    plt.savefig(save_to + "data_dist.png", bbox_inches='tight', dpi=300)
    plt.close()


def make_loss_plot(path_to_dict, save_to):
    """
    Provide results dict to create a
    training validation loss per epoch plot
    :param dict:
    """
    file_to_read = open(path_to_dict, "rb")
    loaded_dict = load(file_to_read)

    train_loss = []
    test_loss = []

    for tensor in loaded_dict['train_loss']:
        if isinstance(tensor, torch.Tensor):
            train_loss.append(tensor.cpu().item())
        else:
            train_loss.append(tensor)
    for tensor in loaded_dict['test_loss']:
        if isinstance(tensor, torch.Tensor):
            test_loss.append(tensor.cpu().item())
        else:
            test_loss.append(tensor)
    plt.plot(train_loss)
    plt.plot(test_loss)
    plt.title('loss per epoch')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train_loss', 'test_loss'], loc='lower left')
    plt.savefig(save_to + "loss.png", bbox_inches='tight', dpi=500)
    plt.close()


def make_loss_plot_fmpn(path_to_dict, save_to):
    pass


def make_acc_plot(path_to_dict, save_to):
    """Provide results dict to create a
    training validation loss per epoch plot"""
    file_to_read = open(path_to_dict, "rb")
    loaded_dict = load(file_to_read)

    train_acc = []
    test_acc = []

    for tensor in loaded_dict['train_acc']:
        if isinstance(tensor, torch.Tensor):
            train_acc.append(tensor.cpu().item())
        else:
            train_acc.append(tensor)
    for tensor in loaded_dict['test_acc']:
        if isinstance(tensor, torch.Tensor):
            test_acc.append(tensor.cpu().item())
        else:
            test_acc.append(tensor)
    plt.plot(train_acc)
    plt.plot(test_acc)
    plt.title('Accuracy per epoch')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train accuracy', 'test accuracy'], loc='lower right')
    plt.savefig(save_to + "acc.png", bbox_inches='tight', dpi=500)
    plt.close()
    # plt.show()


def make_lr_plot(path_to_dict, save_to):
    file_to_read = open(path_to_dict, "rb")
    loaded_dict = load(file_to_read)

    plt.plot(loaded_dict['lr'])
    plt.title('Learning rate per epoch')
    plt.ylabel('learning rate')
    plt.xlabel('epoch')
    plt.legend(['lr'], loc='upper left')
    # plt.show()
    plt.savefig(save_to + "lr.png", bbox_inches='tight', dpi=500)
    plt.close()


def make_lr_plot_fmpn(path_to_dict, save_to):
    lrs = ['lr_fmg', 'lr_pfn', 'lr_cn']
    file_to_read = open(path_to_dict, "rb")
    loaded_dict = load(file_to_read)

    for lr in lrs:
        plt.plot(loaded_dict[lr])
    plt.title('Learning rates per epoch')
    plt.ylabel('learning rate')
    plt.xlabel('epoch')
    plt.legend(['lr_fmg', 'lr_pfn', 'lr_cn'], loc='upper left')
    plt.savefig(save_to + "lr.png", bbox_inches='tight', dpi=500)
    plt.close()


def make_cnfmat_plot(labels, predictions, n_classes, path, gpu_device, title="", dataset="ckp", classnames=None):
    """
    :param labels: torch.tensor of true labels/targets
    :param predictions: torch.tensor of predictions
    :param n_classes: number of classes
    :param path: destination path of the confusion matrix
    :param gpu_device:
    :return:
    """
    cnfmat = torch.zeros(n_classes, n_classes, dtype=torch.int32)
    stack = torch.stack((labels, predictions), dim=1)
    for pair in stack:
        label, pred = pair.tolist()
        cnfmat[int(label), int(pred)] = cnfmat[int(label), int(pred)] + 1
    print(cnfmat)
    np.savetxt(path + "cnfmat.txt", cnfmat.numpy())
    cnfmat = cnfmat.detach().cpu().numpy()
    # calculate the sum of row i over all columns (predictions j)
    # leads to a tensor of shape (n_classes, 1) with
    # num_examples per class i
    cnfmat_sum = cnfmat.sum(axis=1)[:, np.newaxis]
    print(cnfmat_sum)
    for i in cnfmat_sum:
        if i[0] == 0:
            i[0] = 1
    print(cnfmat_sum)
    # if not int(cnfmat_sum):
    #     # if a class is not in the test set it would lead to division by zero
    #     cnfmat_sum = 1
    cnfmatnorm = cnfmat.astype('float') / cnfmat_sum

    if dataset == "ckp":
        cnfmat_df = pd.DataFrame(cnfmatnorm,
                                 index=["anger", "contempt", "disgust", "fear", "happy", "sadness", "surprise"],
                                 columns=["anger", "contempt", "disgust", "fear", "happy", "sadness", "surprise"])
    #elif dataset == "fer" and classnames is not None:
    #    cnfmat_df = pd.DataFrame(cnfmatnorm,
    #                             index=classnames,
    #                             columns=classnames)
    elif dataset != "ckp" and classnames is not None:
        cnfmat_df = pd.DataFrame(cnfmatnorm, index=classnames, columns=classnames)

    ax = sn.heatmap(cnfmat_df, annot=True, annot_kws={"size": 10}, linewidths=5, fmt='.0%', cmap='Blues')
    plt.title(title, fontsize=12)
    plt.xlabel('prediction')
    plt.ylabel('label')
    plt.yticks(rotation=0)
    plt.savefig(path + "cnfmat_plot.png", bbox_inches='tight', dpi=500)
    plt.close()
    return cnfmat
    # plt.show()


def prec_recall_fscore(y_true, y_pred):
    return metrics.precision_recall_fscore_support(y_true=y_true, y_pred=y_pred)


def roc_auc_score(y_true, y_pred, n_classes):
    """
    :param y_true: pytorch.Tensor true labels
    :param y_pred: pytorch.Tensor of predictions
    :return:
    """
    y_true = one_hot(y_true.cpu().long(), num_classes=n_classes).numpy()
    y_pred = one_hot(y_pred.cpu().long(), num_classes=n_classes).numpy()
    return metrics.roc_auc_score(y_true, y_pred, multi_class='ovr', average=None)
