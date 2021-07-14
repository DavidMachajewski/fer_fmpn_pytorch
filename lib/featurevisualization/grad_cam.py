"""
Sources:
Paper:
https://arxiv.org/pdf/1610.02391.pdf
https://arxiv.org/pdf/1512.04150.pdf
GitHub: https://github.com/jacobgil/pytorch-grad-cam

InceptionNet v3: https://github.com/pytorch/vision/blob/master/torchvision/models/inception.py
"""

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision.models import inception_v3
from lib.agents.fmpn_agent import FmpnAgent
from args2 import Setup
import torch


class GradCAMAgent():
    def __init__(self, model, target_layer_nr, use_cuda=False, reshape_transofrm=None):
        if isinstance(model, FmpnAgent):
            self.fmpn_agent = model
            self.model = self.fmpn_agent.cn
            self.ckp_test_dl = self.fmpn_agent.test_dl
        else:  # if you feed e.g. inceptionNet directly
            self.model = model


        self.cam = GradCAM(model=self.model,
                           target_layer=self.get_layer(target_layer_nr),
                           use_cuda=use_cuda,
                           reshape_transform=reshape_transofrm)

    def get_batch(self):
        return next(iter(self.ckp_test_dl))

    def get_layer(self, target_layer_nr):
        return list(self.model.children())[target_layer_nr]

    def get_prediction(self, batch):
        if hasattr(self, self.fmpn_agent):
            probabilities, labels, fusion_imgs = self.fmpn_agent.inference(batch)
            return torch.argmax(probabilities, dim=-1), labels, fusion_imgs

    def create_cams(self):
        batch = self.get_batch()
        images = batch["image"]

        classifications, labels, input_tensor = self.get_prediction(batch)

        grayscale_cam = self.cam(input_tensor=input_tensor,
                                 target_category=None,  # use highest scoring category for each img in batch
                                 aug_smooth=True,
                                 eigen_smooth=True)

        for idx, grayscale_cam_img in enumerate(grayscale_cam):
            visualization = show_cam_on_image(images, grayscale_cam_img)
            #
            # plot this image
            #



if __name__ == "__main__":
    args = Setup().parse()

