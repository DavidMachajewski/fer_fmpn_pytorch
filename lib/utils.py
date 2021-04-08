import numpy as np
import matplotlib.pylab as plt


def imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5
    npimg = img.numpy()
    print("shape: ", np.shape(npimg))
    if one_channel:
        plt.imshow(npimg, cmap="turbo")  # turbo, jet, hot
        plt.show()
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()


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