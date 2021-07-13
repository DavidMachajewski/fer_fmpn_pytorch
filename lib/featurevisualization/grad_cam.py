"""
Sources:
Paper:
https://arxiv.org/pdf/1610.02391.pdf
https://arxiv.org/pdf/1512.04150.pdf
GitHub: https://github.com/jacobgil/pytorch-grad-cam

InceptionNet v3: https://github.com/pytorch/vision/blob/master/torchvision/models/inception.py
"""

# from pytorch_grad_cam import GradCAM
# from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision.models import inception_v3
from lib.agents.fmpn_agent import FmpnAgent


def create_cams():
    model = inception_v3(pretrained=True)
    indexes = [0, 1, 2, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14]
    for name, layer in model.named_children():
        print(name)
        print(layer)


if __name__ == "__main__":
    create_cams()
