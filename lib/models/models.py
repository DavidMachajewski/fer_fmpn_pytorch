import os
import pickle
import re

from tqdm import tqdm
import time
import torch as to
from abc import ABC, abstractmethod
from torch import nn, Tensor
from torch.optim import lr_scheduler
from collections import OrderedDict
import torchvision as tv
from lib.dataloader.datasets import get_ckp
import scipy
import numpy as np
import torch.nn.functional as F



class BuildingBlock(nn.Module):
    """Implementation of a building block for ResNet18/34.
    ResNet50 and higher are using BottleneckBlocks."""
    def __init__(self,
                 inp_filters: int,
                 out_filters: int,
                 stride: int = 1,
                 kernel: int = 3,
                 padding: int = 1):
        super(BuildingBlock, self).__init__()
        self.inp_filters = inp_filters
        self.out_filters = out_filters
        self.kernel = kernel
        self.std_stride = 1
        self.stride = stride
        self.stdnum_channels = 64
        self.padding = padding
        self.convx_1 = nn.Conv2d(self.inp_filters, self.out_filters, self.kernel, self.stride, self.padding, bias=False)
        self.convx_2 = nn.Conv2d(self.out_filters, self.out_filters, self.kernel, self.std_stride, self.padding, bias=False)
        self.bn_1 = nn.BatchNorm2d(self.out_filters)
        self.bn_2 = nn.BatchNorm2d(self.out_filters)
        self.downsample = nn.Conv2d(self.inp_filters, self.out_filters, stride=self.stride, kernel_size=1)
        self.ReLU = nn.ReLU(inplace=True)

    def forward(self, input: Tensor) -> Tensor:
        identity = input
        output = self.convx_1(input)
        output = self.bn_1(output)
        output = self.ReLU(output)
        output = self.convx_2(output)
        output = self.bn_2(output)
        # map crossing of blocks with different filter sizes
        if self.stride == 2:
            identity = self.downsample(input)
        output += identity
        return self.ReLU(output)


class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()
        self.resnet = nn.Sequential(
            nn.Conv2d(3, 64, (7, 7), (2, 2), (3, 3), bias=False),
            nn.MaxPool2d((3, 3), 2, padding=1),
            BuildingBlock(64, 64),
            BuildingBlock(64, 64),
            BuildingBlock(64, 128, stride=2),
            BuildingBlock(128, 128),
            BuildingBlock(128, 256, stride=2),
            BuildingBlock(256, 256),
            BuildingBlock(256, 512, stride=2),
            BuildingBlock(512, 512),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Dropout2d(p=0.5, inplace=True),
            nn.Flatten(1),
            nn.Linear(512, 7),
            nn.Softmax()
        )

    def forward(self, input):
        return self.resnet(input)


class FacialMaskGenerator(nn.Module):
    """Implementation of the Facial Mask Generator (FMG).
    Take grayscaled face images and its corresponding
    emotion ground truth masks to learn a high density map
    per emotion"""
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            # downscaling input image
            nn.Conv2d(1, 64, (7, 7), (1, 1), (3, 3)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, (4, 4), (2, 2), (1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, (4, 4), (2, 2), (1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True))
        self.blocks = [BuildingBlock(256, 256) for i in range(4)]
        self.resblocks = nn.Sequential(*self.blocks)
        self.decoder = nn.Sequential(
            # upscaling input image
            nn.ConvTranspose2d(256, 128, (4, 4), (2, 2), (1, 1), (1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, (4, 4), (2, 2), (1, 1), (1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        self.top = nn.Conv2d(64, 1,
                             kernel_size=(7, 7),
                             stride=(1, 1),
                             padding=(3, 3),
                             bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        input = self.encoder(input)
        input = self.resblocks(input)
        input = self.decoder(input)
        input = self.top(input)
        input = self.sigmoid(input)
        return input


class PriorFusionNetwork(nn.Module):
    """Implementation of the prior fusion network (PFN)
    which prepares original input image and mask image
    for feed to the classification network (CN)"""
    def __init__(self):
        super().__init__()
        # self.shape_fmg = (1, self.args.final_size, self.args.final_size)
        # self.shape_org = (3, self.args.final_size, self.args.final_size)
        self.shape_fmg = (1, 299, 299)
        self.shape_org = (3, 299, 299)
        self.prep_heat = nn.Sequential(
            # prepare multiplied heat map image
            nn.Conv2d(in_channels=self.shape_fmg[0],
                      out_channels=self.shape_org[0],
                      kernel_size=(3, 3),
                      padding=(1,1)),
            nn.BatchNorm2d(self.shape_org[0]),
            nn.ReLU(inplace=True)
        )
        self.prep_org = nn.Sequential(
            # prepare original image
            nn.Conv2d(in_channels=self.shape_org[0],
                      out_channels=self.shape_org[0],
                      kernel_size=(3, 3),
                      padding=(1, 1)),
            nn.BatchNorm2d(self.shape_org[0]),
            nn.ReLU(inplace=True)
        )
        self.resblock = BuildingBlock(self.shape_org[0], self.shape_org[0])
        self.model = nn.Sequential(
            self.resblock,
            nn.Conv2d(in_channels=self.shape_org[0],
                      out_channels=self.shape_org[0],
                      kernel_size=(3, 3),
                      padding=(1, 1)),
            nn.Sigmoid()
        )

    def forward(self, imgorg, imgheat):
        """
        :param img_org: Original image passed to the "FMPN" with shape (3;W;H)
        :param img_heat: Grayscale version of img_org multiplied with predicted gtm by fmg
        :return: img_org + img_mul after some convolutions and
        fitting to the final_shape for classifiers
        """
        imgorg = self.prep_org(imgorg)
        imgheat = self.prep_heat(imgheat)
        #
        # added imgorg and imgheat to return for debugging
        #
        return self.model(imgorg + imgheat), imgorg, imgheat


class NetworkSetup(object):
    """Base Class for setting up networks"""
    def __init__(self, args):
        self.args = args
        self.sub_models = []
        # self.optimizer_fmg = None  # optim.SGD
        # self.optimizer_cn = None  # optim.SGD

        print("cuda:%d device available..." % self.args.gpu_id if to.cuda.is_available() else "cpu")

        self.device = to.device("cuda:%d" % self.args.gpu_id if to.cuda.is_available() else "cpu")

        print(to.cuda.is_available())
        print(to.cuda.current_device())

        self.network_mode = self.args.mode

    @abstractmethod
    def setup(self):
        if self.network_mode == "train":
            print("Setting up network...")
            # set losses (criterions)
            self.calc_loss_fmg = nn.MSELoss()
            self.calc_loss_cn = nn.CrossEntropyLoss()
            self.calc_loss_fmg.to(self.device)
            self.calc_loss_cn.to(self.device)
            to.nn.DataParallel(self.calc_loss_cn, [self.args.gpu_id])
            to.nn.DataParallel(self.calc_loss_fmg, [self.args.gpu_id])
            self.loss_total = lambda loss_fmg, loss_cn: 10 * loss_fmg + 1 * loss_cn
            self.sub_losses = []
            self.optimizers = []
            self.schedulers = []
        else:
            self.set_eval()

    def set_train(self):
        for name in self.sub_models:
            if isinstance(name, str):
                net = getattr(self, name)
                net.train()  # set nn.Module to train state
        self.network_mode = "train"

    def set_eval(self):
        for name in self.sub_models:
            if isinstance(name, str):
                net = getattr(self, name)
                net.eval()
        self.network_mode = "test"

    @abstractmethod
    def feed(self, batch):
        pass

    @abstractmethod
    def forward(self):
        pass

    @abstractmethod
    def backward(self):
        pass

    @abstractmethod
    def optimizer_step(self):
        pass

    def update_learning_rate(self):
        for scheduler in self.schedulers:
            scheduler.step()
        lr = self.optimizers[0].param_groups[0]['lr']
        return lr

    @abstractmethod
    def save(self, epoch, model_names, save_path):
        for name in model_names:
            if isinstance(name, str):
                save_filename = '%s_net_%s.pth' % (epoch, name)
                #
                # ckpt_dir to args !!! within the result folder!
                #
                # save_path = os.path.join(self.args.ckpt_dir, save_filename)
                # save_path = "../../results/" + save_filename
                save_path = save_path + save_filename
                net = getattr(self, name)
                # save cpu params, so that it can be used in other GPU settings
                if to.cuda.is_available():
                    to.save(net.module.cpu().state_dict(), save_path)
                    net.to(self.args.gpu_id)
                    net = to.nn.DataParallel(net, [self.args.gpu_id])
                else:
                    to.save(net.cpu().state_dict(), save_path)

    @abstractmethod
    def load(self, epoch, model_names):
        """
        :param epoch: epoch checkpoints to load
        :param model_names: array of names
        :return:
        """
        for name in model_names:
            if isinstance(name, str):
                load_filename = '%s_net_%s.pth' % (epoch, name)
                load_path = os.path.join(self.args.load_model_dir, load_filename)
                assert os.path.isfile(load_path), "File '%s' does not exist." % load_path

                pretrained_state_dict = to.load(load_path, map_location=str(self.device))
                if hasattr(pretrained_state_dict, '_metadata'):
                    del pretrained_state_dict._metadata

                net = getattr(self, 'net_' + name)
                if isinstance(net, to.nn.DataParallel):
                    net = net.module
                # load only existing keys
                pretrained_dict = {k: v for k, v in pretrained_state_dict.items() if k in net.state_dict()}
                net.load_state_dict(pretrained_dict)
                print("[Info] Successfully load trained weights for %s." % name)

    def get_latest_losses(self, losses_name):
        errors_ret = OrderedDict()
        for name in losses_name:
            if isinstance(name, str):
                cur_loss = float(getattr(self, 'loss_' + name))
                cur_loss_lambda = 1. if len(losses_name) == 1 else float(getattr(self.args, 'lambda_' + name))
                errors_ret[name] = cur_loss * cur_loss_lambda
        return errors_ret


class FMG(NetworkSetup):
    """Wrapper (setup) class for the Facial Mask Generator.

    Pretraining procedure:
    The Facial Mask Generator needs to be trained at first for 300 epochs
    using the Adam optimizer. Initialize learning rate with 0.0001.
    After 150 epochs apply linear decay to 0.

    Joint training procedure:
    Reset the learning rate to 0.00001. Other models use learning rates of 0.0001.
    Jointly training runs for another 200 epochs with linear decay after epoch 100.
    """
    def __init__(self, args):
        super(FMG, self).__init__(args)
        self.fmg = FacialMaskGenerator()
        self.sub_models.append('fmg')

    def setup(self):
        super(FMG, self).setup()
        if self.network_mode == "train":
            self.sub_losses.append("fmg")
            self.optimizer_fmg = to.optim.Adam(
                params=self.fmg.parameters(),
                lr=self.args.lr,
                betas=(self.args.beta1, 0.999))
            self.schedulers.append(get_scheduler(self.optimizer_fmg, self.args))
        if self.args.load_epoch > 0:
            self.load(self.args.load_epoch)

    def feed(self, batch):
        self.images_gray = batch['images_gray']
        self.images_gray.to(self.device)
        if self.network_mode == "train":
            self.masks = batch['mask']
            self.masks.to(self.device)

    def forward(self):
        self.predicted_mask = self.fmg(self.images_gray)

    def backward(self):
        self.loss_fmg = self.calc_loss_fmg(self.predicted_mask, self.masks)
        self.loss_fmg.backward()

    def optimizer_step(self):
        self.forward()
        self.optimizer_fmg.zero_grad()
        self.backward()
        self.optimizer_fmg.step()

    def load(self, epoch):
        models = ['fmg']
        return super(FMG, self).load(epoch, models)

    def save(self, epoch, savepath):
        models = ['fmg']
        return super(FMG, self).save(epoch, models, savepath)

    def get_latest_losses(self, losses_name):
        losses_names = ['fmg']
        return super(FMG, self).get_latest_losses(losses_names)


class FMPN(NetworkSetup):
    def __init__(self, args):
        super(FMPN, self).__init__(args)
        self.fmg = FacialMaskGenerator()
        self.fmg.to(self.device)
        self.fmg = to.nn.DataParallel(self.fmg, [self.device])
        self.pfn = PriorFusionNetwork()
        self.pfn.to(self.device)
        self.pfn = to.nn.DataParallel(self.pfn, [self.device])
        self.cn = tv.models.Inception3(num_classes=7, init_weights=True)  # get classification network
        self.cn.to(self.device)
        self.cn = to.nn.DataParallel(self.cn, [self.device])
        self.sub_models.append("fmg")
        self.sub_models.append("pfn")
        self.sub_models.append("cn")

    def setup(self):
        super(FMPN,self).setup()
        if self.network_mode == "train":
            self.sub_losses.append("fmg")
            self.sub_losses.append("cn")
            self.optmizer_fmpn = to.optim.Adam(
                params=
                [
                    {'params': self.fmg.parameters()},
                    {'params': self.pfn.parameters()},
                    {'params': self.cn.parameters(),
                     'lr': self.args.lr_fmg}
                ],
                lr=self.args.lr,
                betas=(self.args.beta1, 0.999)
            )
            self.optimizers.append(self.optmizer_fmpn)
            self.schedulers.append(get_scheduler(self.optmizer_fmpn, args=self.args))
        if self.args.load_epoch > 0:
            self.load(self.args.load_epoch)

    def feed(self, batch):
        self.images, self.images_gray = batch["image"], batch["image_gray"]
        self.images = self.images.to(self.device)
        self.images_gray = self.images_gray.to(self.device)
        # print("FEED IMAGES GRAY IS CUDA: ", self.images_gray.is_cuda)
        if self.network_mode == "train":
            self.labels, self.masks = batch["label"], batch["mask"]
            self.labels = self.labels.type(to.LongTensor).to(self.device)
            self.masks = self.masks.to(self.device)

    def forward(self):
        self.predicted_mask = self.fmg(self.images_gray)
        self.images_gray.to(self.device)
        self.heat_face = self.predicted_mask * self.images_gray
        self.fusion_imgs = self.pfn(self.images, self.heat_face)
        self.predicted_class = self.cn(self.fusion_imgs)

    def backward(self):
        # print("Running backward path: \n")
        self.loss_fmg = self.calc_loss_fmg(self.images_gray, self.predicted_mask)
        # print("loss_fmg: ", self.loss_fmg)
        # print("shapes: ", self.labels)
        # self.predicted_class = to.argmax(self.predicted_class.logits, dim=-1)
        # print("shape_2: ", self.predicted_class)
        # self.labels = F.one_hot(self.labels, num_classes=7)
        # print("labels: ", self.labels)

        self.loss_cn = self.calc_loss_cn(self.predicted_class.logits, self.labels)
        # print("loss_cn: ", self.loss_cn)
        self.total_fmpn_loss = self.loss_total(self.loss_fmg, self.loss_cn)
        # print("total loss: ", self.total_fmpn_loss)
        self.total_fmpn_loss.backward()


    def optimizer_step(self):
        self.forward()
        self.optmizer_fmpn.zero_grad()
        self.backward()
        self.optmizer_fmpn.step()

    def load(self, epoch):
        models = ['fmg']
        if not self.network_mode == "train":
            models.extend(['pfn', 'cn'])
        return super(FMPN, self).load(epoch, models)

    def save(self, epoch, path):
        models = ['fmg', 'pfn', 'cn']
        return super(FMPN, self).save(epoch, models, path)

    def get_latest_losses(self):
        losses_names = ['fmg', 'cn']
        return super(FMPN, self).get_latest_losses(losses_names)

    def get_latest_accuracy(self):
        pass


class RunSetup(object):
    def __init__(self, args):
        self.args = args
        self.timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        self.model_to_train = self.args.model_to_train
        self.model = None
        self.run_path = None
        self.train_path = None
        self.test_path = None
        self.ckpt = None
        self.train_plots = None
        self.test_plots = None
        self.paths_dict = None
        self.train = None
        self.test = None

        self.gpu = to.device("cuda:%d" % self.args.gpu_id if to.cuda.is_available() else "cpu")

        self.create_folders()

    def init_model(self, model):
        self.model = model
        self.model.setup()
        # self.model.to(self.gpu)

    def load_data(self, train, test):
        self.train = train
        self.test = test

    def train_loop(self, epochs):
        lr_per_epoch = []
        loss_per_epoch = []

        for epoch in range(epochs):
            with tqdm(self.train, unit="batch") as tepoch:
                for i, batch in enumerate(tepoch):
                    tepoch.set_description(f"Epoch {epoch}")
                    self.model.feed(batch)
                    self.model.optimizer_step()
                    tmp_loss = self.model.get_latest_losses()
                    cur_lr = self.model.update_learning_rate()
                    tepoch.set_postfix(loss=tmp_loss, accuracy="n.a.")
            loss_per_epoch.append(tmp_loss)
            lr_per_epoch.append(cur_lr)
        info_dict = {
            'model': self.model_to_train,
            'epochs_total': epochs,
            'loss_per_epoch': loss_per_epoch,
            'lr_per_epoch': lr_per_epoch
        }
        filte_to_write = open(self.train_path + "info_dict.pickle", "wb")
        pickle.dump(info_dict, filte_to_write)
        self.model.save(epochs, self.train_path)

    def test_loop(self):
        pass

    def create_folders(self):
        if not os.path.isdir(self.args.result_folder):
            os.makedirs(self.args.result_folder)
        print(type(self.timestamp))
        print(type(self.args.result_folder))
        self.run_path = self.args.result_folder + "run_" + self.model_to_train + "_{0}".format(self.timestamp) + "/"
        print(self.run_path)
        os.makedirs(self.run_path)
        self.train_path = self.run_path \
                     + "train_" + self.model_to_train + "_" \
                     + self.timestamp + "/"
        self.test_path = self.run_path \
                     + "test_" + self.model_to_train + "_" \
                     + self.timestamp + "/"
        os.makedirs(self.train_path)

        os.makedirs(self.train_path + "ckpt/")
        os.makedirs(self.train_path + "plots/")
        os.makedirs(self.test_path)
        os.makedirs(self.test_path + "plots/")


class RunFMG(RunSetup):
    def __init__(self, args):
        self.args = args
        self.args.mode_to_train = "fmg"
        super(RunFMG, self).__init__(self.args)

    def start(self):
        self.init_model()
        self.load_data()
        self.train_loop()

    def init_model(self):
        model = FMG(self.args)
        super(RunFMG, self).init_model(model)

    def load_data(self):
        if self.args.dataset == "ckp":
            train_ds, test_ds = get_ckp(args=self.args)
            train_ds.to(self.gpu)
            test_ds.to(self.gpu)
            super(RunFMG, self).load_data(train_ds, test_ds)

    def train_loop(self):
        super(RunFMG, self).train_loop(self.args.epochs)

    def test_loop(self):
        pass


class RunFMPN(RunSetup):
    def __init__(self, args):
        self.args = args
        self.args.model_to_train = "fmpn"
        super(RunFMPN, self).__init__(self.args)

    def start(self):
        self.init_model()
        self.load_data()
        self.train_loop()

    def init_model(self):
        model = FMPN(self.args)
        super(RunFMPN, self).init_model(model)

    def load_data(self):
        if self.args.dataset == "ckp":
            train_ds, test_ds = get_ckp(args=self.args)
        super(RunFMPN, self).load_data(train_ds, test_ds)

    def train_loop(self):
        super(RunFMPN, self).train_loop(self.args.epochs)

    def test_loop(self):
        #
        #
        #
        pass



def vgg19(pretrained=False):
    """Source of vgg pytorch:
    https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py"""
    n_channels = 1
    n_classes = 7
    if pretrained:
        print("Loading pretrained model...")
        vgg = tv.models.vgg19(transform_input=True, init_weights=False)
        state_dict = to.hub.load_state_dict_from_url(
            'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth')
        state = {k: v for k, v in state_dict.items() if k in vgg.state_dict()}
        vgg.load_state_dict(state)
        print("Loaded ")
        vgg.features[0] = to.nn.Conv2d(n_channels, 64, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        vgg.classifier[6] = to.nn.Linear(in_features=4096, out_features=n_classes, bias=True)
    else:
        print("Loading non pretrained model...")
        vgg = tv.models.vgg19(transform_input=True, init_weights=False)
        vgg.features[0] = to.nn.Conv2d(n_channels, 64, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        vgg.classifier[6] = to.nn.Linear(in_features=4096, out_features=n_classes, bias=True)
    return vgg


def inceptionv3(pretrained=False, n_classes=7):
    """Input size for inception net is (3. 229, 299) """
    if pretrained:
        print("Loading pretrained model...")
        inc = tv.models.Inception3(transform_input=True,
                                   init_weights=False)
        state_dict = to.hub.load_state_dict_from_url(
            'https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth')
        state = {k: v for k, v in state_dict.items() if k in inc.state_dict()}
        inc.load_state_dict(state)
        print("Loaded ")
        #
        # :TODO: flexible number of classes! args.n_classes
        #
        inc.fc = nn.Linear(in_features=2048, out_features=n_classes)
        to.nn.init.xavier_normal_(inc.fc.weight.data, gain=0.02)
        to.nn.init.constant_(inc.fc.bias.data, 0.0)
    else:
        print("Loading non pretrained model...")
        inc = tv.models.Inception3(num_classes=n_classes)
        inc.fc = nn.Linear(in_features=2048, out_features=n_classes)
    return inc


def densenet121(args, pretrained=False):
    if pretrained:
        print("Initialize pretrained model.")
        """http://data.lip6.fr/cadene/pretrainedmodels/densenet121-fbdb23505.pth"""
        dense = tv.models.densenet121(pretrained=pretrained,
                                      # init_weights=False,
                                      # growth_rate=args.dn_growth,
                                      # num_init_features=args.dn_features,
                                      # bn_size=args.dn_bnsize
                                      )
        state_dict = to.hub.load_state_dict_from_url('http://data.lip6.fr/cadene/pretrainedmodels/densenet121-fbdb23505.pth')
        pattern = re.compile(
            r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
        for key in list(state_dict.keys()):
            res = pattern.match(key)
            if res:
                new_key = res.group(1) + res.group(2)
                state_dict[new_key] = state_dict[key]
                del state_dict[key]
        dense.load_state_dict(state_dict)
        print("Loaded ")
        # dense.fc = nn.Linear(2048, 7)
        dense.classifier = nn.Linear(1024, 7)
        to.nn.init.xavier_normal_(dense.classifier.weight.data, gain=0.02)
        to.nn.init.constant_(dense.classifier.bias.data, 0.0)
    else:
        print("Inititialize non pretrained model.")
        dense = tv.models.densenet121(num_classes=7  #,
                                      #growth_rate=args.dn_growth,
                                      #num_init_features=args.dn_features,
                                      #bn_size=args.dn_bnsize
                                      )
    return dense


#
#  SCHEDULER
#
def get_scheduler(optimizer, args):
    def lambda_rule(epoch):
        if epoch >= args.start_lr_drop:  # 150
            lr_l = 1.0
        lr_l = 1.0 - max(0, epoch + 1 + args.epoch_count - args.niter) / float(args.niter_decay + 1)
        return lr_l

    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    return scheduler