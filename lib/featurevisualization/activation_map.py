"""
Class Activation Map
https://medium.com/intelligentmachines/implementation-of-class-activation-map-cam-with-pytorch-c32f7e414923

Learning Deep Features for Discriminative Localization
https://arxiv.org/pdf/1512.04150.pdf


- class activation map for a particular category indicates the particular
region used by cnn to identify the output class

(1) perform global average pooling before the final output layer
(2) resulting features are fed to a fully connected layer with softmax activation
(3) project weights of the output layer back into the convolutionary maps
    derived from the last convolution layer


Implementation
(1) Load dataset/dataloader and a model like inceptionNet/VGG/...
(2) Extract all layers besides the last fully connected layer of the chosen model
    InceptionNet implementation of pytorch: https://github.com/pytorch/vision/blob/master/torchvision/models/inception.py
(3) create a model that transforms the last convolutional outputs dimensions
    to an output of n_classes features where n_classes is the number of classes.
    Last convolutional layer output of Inception_v3 net is of dimension 8 x 8 x 2048
    This needs to be converted to 2048 x 64 and then to 2045 x 1. Then convert to 1 x 2048
    and apply a Linear Layer which takes all this 2048 features and gives n_classes output features.
(4) Add ModifyInceptionv3 to the extracted model from step (3)


"""
import torch
import numpy as np
import torch.nn.functional as F
from lib.agents.fmpn_agent import FmpnAgent
from lib.agents.runner import Runner


class ModifyInceptionv3(torch.nn.Module):
    def __init__(self, in_features, args):
        self.args = args
        self.in_features = in_features
        super(ModifyInceptionv3, self).__init__()
        self.fc = torch.nn.Linear(in_features=in_features,
                                  out_features=self.args.n_classes)

    def forward(self, input):
        # transform input tensor
        input = input.view(self.in_features, 8*8).mean(1).view(1, -1)
        input = input.fc(input)
        return F.softmax(input, dim=1)


def get_modified_model(model, args):
    """extract last last layer and modify e.g. inceptionNet"""
    extracted = torch.nn.Sequential(*list(model.children())[:-1])
    return torch.nn.Sequential(extracted, ModifyInceptionv3(2048, args))


def init_class_activation_mapping(runner: Runner):
    if runner.args.model_to_train == "fmpn":
        # get classification netwok (pretrained inceptionNet in this case)
        runner.model.cn = get_modified_model(runner.model.cn, runner.args)
    return runner


def create_class_activation_maps(agent: FmpnAgent, map_size=(256, 256)):
    """restore the modified and trained model"""
    n_classes = agent.args.n_classes

    parameters = list(ModifyInceptionv3().parameters())
    weights = np.squeeze(parameters[-1].data.numpy())












