# ####################################
#
# Source:
# ####################################
from PIL import Image, ImageFilter, ImageChops
import numpy as np
import torch
import scipy.ndimage as nd
from lib.models.models import inceptionv3
from lib.agents.inc_agent import InceptionAgent
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision import transforms


class DeepDream:
    """Implementation of the DeepDream Algorithm
    by A. Mord..
    Soruces: (1) https://github.com/google/deepdream/blob/master/dream.ipynb
             (2) https://web.archive.org/web/20150708233755/http://googleresearch.blogspot.co.uk/2015/07/icml-2015-and-machine-learning-research.html

    1) Load trained model
    2) Gradient ascent process to maximize the l2 norm of
       activations of a particular network layer
       a) gradient ascent step function
         i) offset image by a random jitter
         ii) normalize the magnitude of gradient ascent steps
       b) ascent trough different scales (octaves)

    We choose the Layer approach in case of optimization objectives
    """

    def __init__(self, args):
        self.args = args
        self.model, self.train_dl, self.test_dl = self.load_model()
        self.layer_names = self.__get_layer_names__()
        print(self.layer_names)
        self.activation = {}
        self.images_used = []

        self.denormalize = transforms.Compose(
            [transforms.Normalize(mean=[0., 0., 0.], std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
             transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1., 1., 1.])])

    def get_activation(self, name):
        def hook(model, input, output):
            # print("make a hook!")
            # self.activation[name] = output.detach()
            self.activation[name] = output
        return hook

    def __get_img_sample__(self, plot: bool = False):
        batch = next(iter(self.train_dl))
        img = batch["image"].squeeze()
        label = batch["label"]
        if plot:
            plt.imshow(img.permute(1, 2, 0))
            plt.show()
        return img, label

    def __get_layer_names__(self):
        layer_names = []
        for name, layer in self.model.named_modules():
            #if isinstance(layer, torch.nn.Conv2d):
            layer_names.append(name)
        return layer_names

    def load_model(self):
        """Load and restore inception agent with a checkpoint
        and return the inception model.

        following must be provided by the args:
        pretrained 1 -> pretrained on imagenet?
        load_ckpt 1 -> if 1 provide ckpt_to_load
        ckpt_to_load -> inception agent checkpoints
        """
        deepdream_model = self.args.deepdream_model
        if deepdream_model == "incv3":
            agent = InceptionAgent(args=self.args)
            return agent.model, agent.train_dl, agent.test_dl

    def gradient_step(self, input_image):
        """
        :param input_image: image of type tensor
        :return:
        """
        with torch.enable_grad():
            # input_image = input_image.to(torch.device('cuda:0'))
            # input_image.requires_grad = True
            self.model.zero_grad()
            print("GRADIENT?: ", input_image[0].requires_grad)
            self.model.Conv2d_1a_3x3.register_forward_hook(self.get_activation('Conv2d_1a_3x3.conv'))
            self.model.Conv2d_2a_3x3.register_forward_hook(self.get_activation('Conv2d_2a_3x3.conv'))
            self.model.Mixed_5b.register_forward_hook(self.get_activation('Mixed_5b.branch1x1.conv'))
            output = self.model(input_image)
            print(len(self.activation))
            print(np.shape(self.activation["Conv2d_1a_3x3.conv"]))

            losses, activations = [], []
            for key in self.activation.keys():
                for tensor in self.activation[key]:
                    activations.append(tensor)

            for tensor in activations:
                lossc = torch.nn.MSELoss(reduction='mean')(tensor, torch.zeros_like(tensor))
                losses.append(lossc)

            loss = torch.mean(torch.stack(losses))
            loss.backward()

            print("Input tensor type: ", type(input_image))
            print("input image of tensor type: ", type(input_image[0]))
            gradient = input_image.grad.data
            print(gradient[0])

            sigma = 1 / 10 * 2 + 2.0 * 0.5

            #smooth_grad =

            # normalize gradients
            for grad in gradient:
                g_std = torch.std(gradient)
                g_mean = torch.mean(gradient)
                gradient = gradient - g_mean
                gradient = gradient / g_std

            # update images using calculated gradients (ascent step)
            for tensor in input_image:
                tensor += 0.1 * gradient


    def __new_shape__(self, base_shape, pyramid_level):
        shape_margin = 10
        pyramid_ratio = 1.8
        pyramid_size = 4
        exponent = pyramid_level - pyramid_size + 1
        new_shape = np.round(np.float32(base_shape) * (pyramid_ratio ** exponent)).astype(np.int32)
        return new_shape


    def dream(self, base_img, iter_n = 10, octave_n = 4, octave_scale = 1.4):
        """
        :param base_img:
        :param iter_n:
        :param octave_n:
        :param octave_scale:
        :return:
        """
        base_imgs = []
        base_shape = base_img[0].shape[1:]

        for image_tensor in base_img:
            base_imgs.append(image_tensor.unsqueeze(0))

        for idx, image_tensor in enumerate(base_imgs):
            for octave_id in range(octave_n):
                new_shape = self.__new_shape__(base_shape, octave_id)
                image_tensor = F.interpolate(image_tensor, size=(new_shape[0], new_shape[1]))
                print(image_tensor.shape)

                for iteration in range(iter_n):
                    h_shift, w_shift = np.random.randint(-32, 32+1, 2)
                    # print("shift: ", h_shift)
                    # print("shift2: ", w_shift)
                    image_tensor = self.jitter_shift(image_tensor, h_shift, w_shift)
                    print("after jitter: ", type(image_tensor))
                    image_tensor_batch = torch.cat((image_tensor, image_tensor.clone()), dim=0)
                    print("after jitter2: ", image_tensor_batch.shape)
                    self.gradient_step(image_tensor_batch)
                    image_tensor = self.jitter_shift(image_tensor, h_shift, w_shift, deshift=True)


    def jitter_shift(self, image_tensor, h_shift, w_shift, deshift=False):
        if deshift:
            h_shift = -h_shift
            w_shift = -w_shift
        with torch.no_grad():
            image_tensor = torch.roll(image_tensor, shifts=(h_shift, w_shift), dims=(2, 3))
            image_tensor.requires_grad = True
            return image_tensor

    def hook_layer_no(self, idx):
        layer_to_grab = [
            self.model.Conv2d_1a_3x3.register_forward_hook(self.get_activation('Conv2d_1a_3x3.conv')),
            self.model.Conv2d_2a_3x3.register_forward_hook(self.get_activation('Conv2d_2a_3x3.conv')),
            self.model.Mixed_5b.register_forward_hook(self.get_activation('Mixed_5b.branch1x1.conv')),
            self.model.Mixed_7a.register_forward_hook(self.get_activation('Mixed_5b.branch1x1.conv')),
            self.model.Mixed_7c.register_forward_hook(self.get_activation('Mixed_7c.branch3x3dbl_2.conv'))
        ]
        return layer_to_grab[idx]

    def get_gradients(self, img_batch, layer_no):
        print("get gradients")

        for tensor in img_batch:
            print(tensor.requires_grad)

        self.model.zero_grad()

        # self.model.Conv2d_1a_3x3.register_forward_hook(self.get_activation('Conv2d_1a_3x3.conv'))
        # self.model.Conv2d_2a_3x3.register_forward_hook(self.get_activation('Conv2d_2a_3x3.conv'))
        # self.model.Mixed_5b.register_forward_hook(self.get_activation('Mixed_5b.branch1x1.conv'))
        # self.model.Mixed_7a.register_forward_hook(self.get_activation('Mixed_5b.branch1x1.conv'))
        # self.model.Mixed_7c.register_forward_hook(self.get_activation('Mixed_7c.branch3x3dbl_2.conv'))
        self.hook_layer_no(layer_no)

        predictions = self.model(img_batch)

        loss = []
        for key in self.activation:
            loss.append(self.activation[key][0].norm())
            # print(self.activation[key])
            #for hook in self.activation[key]:
            #    print(hook[0].norm())

        print("amount of loss: ", loss)
        loss[0].backward()

        print("gradient data of image")
        print(np.shape(img_batch.grad.data))
        return img_batch.grad.data

    def start_dreaming(self, img_batch, iterations_n=20, lr=1.8, layer_no=0):
        print("start dreaming")
        img_batch = img_batch
        for i in range(iterations_n):
            print("Iteration no. {0}".format(i))
            gradients_batch = self.get_gradients(img_batch, layer_no)
            for idx, image_tensor in enumerate(img_batch):
                with torch.no_grad():
                    img_batch[idx] = img_batch[idx] + lr * gradients_batch[idx].squeeze()

        imgs = img_batch.detach().cpu()

        imgs = self.denormalize(imgs)
        imgs_np = []
        for tensor in imgs:
            imgs_np.append(tensor.numpy().transpose(1,2,0))

        imgs_plot = []
        for imgs in imgs_np:
            imgs_plot.append(Image.fromarray(np.uint8(imgs * 255)))
        return imgs_plot




"""


    def deepdream(self,base_img,iter_n=10,octave_n=4,octave_scale=1.4):
        images = [tensor for tensor in base_img]

        octaves = [[image] for image in images]

        # prepare base images
        shapes = [tensor.size() for tensor in base_img]

        for batch_num in range(len(octaves)):
            for j in range(octave_n - 1):
                zoom_img = torch.from_numpy(nd.zoom(octaves[batch_num][-1].cpu(), (1, 1.0/octave_scale,1.0/octave_scale), order=1))
                octaves[batch_num].append(zoom_img)


        print(len(octaves[0]))
        print(len(octaves[1]))

        detail = np.zeros_like(octaves[0][-1])

"""