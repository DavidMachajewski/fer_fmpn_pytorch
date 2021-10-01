"""
Sources:
Paper:
https://arxiv.org/pdf/1610.02391.pdf
https://arxiv.org/pdf/1512.04150.pdf
GitHub: https://github.com/jacobgil/pytorch-grad-cam

InceptionNet v3: https://github.com/pytorch/vision/blob/master/torchvision/models/inception.py
"""
import os

import numpy as np
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision.models import inception_v3
from lib.agents.fmpn_agent import FmpnAgent
from args2 import Setup
import torch
import matplotlib.pyplot as plt
import cv2 as cv
from datetime import datetime
from lib.dataloader.datasets import DeNormalize

class GradCAMAgent():
    def __init__(self, model, target_layer_nr, use_cuda=False, reshape_transofrm=None):
        self.target_layer_nr = target_layer_nr
        if isinstance(model, FmpnAgent):
            self.fmpn_agent = model
            self.model = self.fmpn_agent.cn
            self.ckp_test_dl = self.fmpn_agent.test_dl
        else:  # if you feed e.g. inceptionNet directly
            self.agent = model
            self.model = self.agent.model  # InceptionAgent
            self.ckp_test_dl = self.agent.test_dl

        self.cam = GradCAM(model=self.model,
                           target_layer=self.get_layer(target_layer_nr),
                           use_cuda=use_cuda,
                           reshape_transform=reshape_transofrm)

    def get_batch(self):
        return next(iter(self.ckp_test_dl))

    def get_layer(self, target_layer_nr):
        print(type(list(self.model.children())[target_layer_nr]))
        return list(self.model.children())[target_layer_nr]

    def get_prediction(self, batch):
        if hasattr(self, "fmpn_agent"):
            probabilities, labels, fusion_imgs = self.fmpn_agent.inference(batch)
            return torch.argmax(probabilities, dim=-1), labels, fusion_imgs
        else:
            probabilities, labels = self.agent.inference(batch)
            return torch.argmax(probabilities, dim=-1), labels

    def get_emotion(self, class_id) -> str:
        emotions = ['anger', 'contempt', 'disgust', 'fear', 'happy', 'sadness', 'surprise']
        return emotions[class_id]

    def get_rd(self):
        now = datetime.now()
        current = now.strftime("%H-%M-%S")
        return current

    def create_cams(self, save_to: str):

        file_name = "gradcam"

        print("creating gradcams ...")
        # batch = self.get_batch()
        # images = batch["image"]

        for batch_id, batch in enumerate(self.ckp_test_dl):
            if hasattr(self, "fmpn_agent"):
                classifications, labels, input_tensor = self.get_prediction(batch)
            else:
                classifications, labels = self.get_prediction(batch)
                input_tensor = batch["image"].cuda()
            # print("classifications: \n", classifications)
            # print("true classes: \n", labels)
            # print("tensor shape: ", input_tensor.shape)

            grayscale_cam = self.cam(input_tensor=input_tensor,
                                     target_category=classifications,  # use highest scoring category for each img in batch
                                     aug_smooth=True,
                                     eigen_smooth=True)

            hstack_images = []
            for idx, grayscale_cam_img in enumerate(grayscale_cam):
                file_name_vis = "gradcam_layer_{0}_batch_{1}_batchimg_{2}_class_{3}_{4}_vis.png".format(self.target_layer_nr, batch_id, idx, labels[idx], self.get_rd())
                file_name = "gradcam_layer_{0}_batch_{1}_batchimg_{2}_class_{3}_{4}.png".format(self.target_layer_nr, batch_id, idx, labels[idx], self.get_rd())
                visualization = show_cam_on_image(input_tensor[idx].cpu().detach().permute(1, 2, 0).numpy(), grayscale_cam_img, True)
                # führe die gleiche visualisierung nochmal für die normalen bilder aus batch["image"]
                visualization = visualization/255.0
                #plt.imshow(visualization)
                #plt.show()

                visualization = visualization * 255.0

                #  print(type(batch["image"][idx]))
                #  batch["image"][idx].mul_([0.229, 0.224, 0.225]).add_([0.485, 0.456, 0.406])

                origimg = batch["image"][idx].detach().permute(1, 2, 0).numpy() * 255.0
                origimg = origimg.astype(np.uint8)

                fusionimg = input_tensor[idx].cpu().detach().permute(1, 2, 0).numpy() * 255.0
                fusionimg = fusionimg.astype(np.uint8)


                hstack_images.append(origimg)
                mask = batch["mask"][idx].detach().permute(1, 2, 0).numpy() * 255.0
                mask_rgb = cv.cvtColor(mask, cv.COLOR_GRAY2RGB).astype(np.uint8)
                hstack_images.append(mask_rgb.astype(np.uint8))
                hstack_images.append(fusionimg)
                hstack_images.append(visualization.astype(np.uint8))

                hstacked_images = np.hstack(tuple(hstack_images))

                # set text on images
                text_layer = "layer nr: {0}, {1}".format(self.target_layer_nr, type(self.get_layer(self.target_layer_nr)))
                text_predicted_emotion = "label: {0}, predicted: {1}".format(self.get_emotion(labels[idx]), self.get_emotion(classifications[idx]))

                cv.putText(hstacked_images,
                           text_layer,
                           (10,int(9.25*30)),
                           cv.FONT_HERSHEY_SIMPLEX,
                           0.78,
                           (255,0,0),
                           2)
                cv.putText(hstacked_images,
                           text_predicted_emotion,
                           (10, int(1 * 30)),
                           cv.FONT_HERSHEY_SIMPLEX,
                           0.78,
                           (255,0,0),
                           2)

                hstacked_images = hstacked_images.astype(np.uint8)

                # save fusioned image (can be used for average creation
                plt.imsave(os.path.join(save_to, file_name_vis), visualization.astype(np.uint8))

                # save stacked image
                plt.imsave(os.path.join(save_to, file_name), hstacked_images)

                # plt.show()
                hstack_images = []


if __name__ == "__main__":
    args = Setup().parse()