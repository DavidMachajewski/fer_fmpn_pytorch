import numpy as np
from torchvision.utils import save_image
import matplotlib.pylab as plt


def imshow_tensor(img, one_channel=False, path=None, show=False):
    """
    :param img:
    :param one_channel:
    :return:
    """
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5
    npimg = img.numpy()
    # print("shape: ", np.shape(npimg))
    if one_channel:
        if path is not None:
            # plt.savefig(saveto)
            plt.imsave(path, npimg, cmap="turbo")
        if show:
            plt.imshow(npimg, cmap="turbo")  # turbo, jet, hot
            plt.show()
    else:
        # plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.imsave(saveto, npimg, cmap="turbo")
        # plt.show()


def save_tensor_img(img, path):
    #img = np.transpose(img, (1, 2, 0))
    save_image(img, path)


"""
Use it like this for example:

batch = next(iter(train))
img_org = batch["image"]
mask = batch["mask"]

# print(batch)
imshow(img_org[0], one_channel=False)
print(np.shape(mask[0]))
imshow(mask[0], one_channel=True)

# imshow(torchvision.utils.make_grid(img_org), one_channel=False)
# imshow(torchvision.utils.make_grid(mask), one_channel=True)
"""