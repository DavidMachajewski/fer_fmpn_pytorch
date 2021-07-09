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
import cv2 as cv


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



class CAMCreator:
    def __init__(self, model: FmpnAgent):
        self.model = model
        self.init()
        self.activation = []

    def init(self):
        self.parameters = list(self.model.cn.fc.parameters())
        self.weights = self.model.cn.fc.weight  # parameters[0] shape (7, 2048)
        # print(self.model.cn.Mixed_7c.branch_pool)
        self.hook_activation(self.model.cn.Mixed_7c)
        self.model.cn.Mixed_7c.register_forward_hook(self.hook_activation(self.model.cn.Mixed_7c))
        self.test_dl = self.model.test_dl

    def hook_activation(self, module):
        def hook(module, input, output):
            # self.activation[name] = output
            self.activation.append(output)
        return hook

    def __create_cam__(feature_map: torch.Tensor,
                       conv_weights: torch.nn.parameter.Parameter,
                       predicted_class,
                       upscale=(256, 256)):
        n_batches, n_channels, height, width = feature_map.shape
        cam_out = []



    def build_map(self, batch, img_id):
        """
        The weights of the last fc layer are of shape (7, 2048)
        The activations have shape (2048, 8, 8) -> reshape to (2048, 64)
        The weights of class x have shape (1, 2048)

        Obtain the feature map of the last convolutional layer
        and the corresponding weights to the highest probability
        of the output layer (predicted class). Now multiply the
        feature map with the weights, resize the heatmap to the
        original image size and sum both.


        :param batch: batch of test_dl
        :return:
        """
        probabilities, labels = self.model.inference(batch)
        probabilities = torch.argmax(probabilities, 1)
        cam_out = []
        # print("First image of batch...")
        # print(batch["image"][0].shape)
        label = batch["label"][0]
        nc_img, height_img, width_img = batch["image"][0].shape


        batch_img = 0
        upscale = (256, 256)
        print("shape of activations of last conv layer: ", self.activation[0].shape)
        print("shape of activation of first image of batch: ", self.activation[0][0].shape)

        activation_reshaped = self.activation[0].cpu()[batch_img].reshape((2048, 64))
        print("shape of reshaped activation of first image of batch: ", activation_reshaped.shape)
        # print(activation_reshaped)
        predicted_class = probabilities.cpu()[batch_img]
        # print(predicted_class)
        print("shape of all weights: ", self.weights.shape)
        weight = self.weights.cpu()[predicted_class]
        print("Weights of class {0} : \n {1}".format(predicted_class, weight))
        print("maximum weight: ", np.max(weight.detach().numpy()))
        print("var: ", np.var(weight.detach().numpy()))
        print("std: ", np.sqrt(np.var(weight.detach().numpy())))
        print("shape: {}".format(weight.shape))
        cam = np.matmul(weight.detach(), activation_reshaped.detach())
        # print(cam.shape)
        cam_reshaped = cam.reshape(8, 8).detach().numpy()

        cam_reshaped = cam_reshaped - np.min(cam_reshaped)
        cam_img = cam_reshaped / np.max(cam_reshaped)
        cam_img = np.uint8(cam_img)
        cam_out.append(cv.resize(cam_img, upscale))

        heatmap = cv.applyColorMap(
            src=cv.resize(cam_out[0], (width_img, height_img)),
            colormap=cv.COLORMAP_JET)
        # heatmap = cv.normalize(heatmap, heatmap, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)
        #cv.imshow('heatmap', heatmap)
        result = (0.004 * heatmap + 1.0 * batch["image"][0].permute(1, 2, 0).detach().numpy())
        #cv.imshow('image', result)
        #cv.waitKey(0)
        orig_img = batch["image"][0].permute(1, 2, 0).detach().numpy()
        #cv.imshow('orig', orig_img)
        #cv.waitKey(0)

        imgs_to_stack = [255 * orig_img, heatmap, 255 * result]
        stacked_images = np.hstack(tuple(imgs_to_stack))
        # result_superimposed = cv.addWeighted(heatmap, 0.7, batch["image"][0].permute(1, 2, 0).detach().numpy(), 0.3, 0)
        # cv.imshow('image_superimp', result_superimposed)
        #cv.imshow('merged: ', stacked_images)

        #cv.waitKey(0)
        text_predicted_emotion = "label: {0}, predicted: {1}".format(self.get_emotion(label), self.get_emotion(predicted_class))
        cv.putText(stacked_images,
                   text_predicted_emotion,
                   (10,int(9.25*30)),
                   cv.FONT_HERSHEY_SIMPLEX,
                   0.75,
                   (0,0,255),
                   2)
        #cv.imshow('with text', stacked_images)
        #cv.waitKey(0)
        cv.imwrite("class_activation_map_{0}.png".format(img_id), stacked_images)

    def get_emotion(self, class_id) -> str:
        emotions = ['anger', 'contempt', 'disgust', 'fear', 'happy', 'sadness', 'surprise']
        return emotions[class_id]
















