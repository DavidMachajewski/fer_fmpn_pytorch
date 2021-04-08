import os
import pandas as pd
import torch as to
from torch.utils.data import Dataset, DataLoader
import pickle
import cv2 as cv
import numpy as np
import torchvision as tv


class DatasetBase(Dataset):
    def __init__(self, args, train: bool):
        self.args = args
        super(DatasetBase, self).__init__()
        self.data_root = self.args.data_root
        self.masks_path = self.args.masks

        if train:
            self.splitname = args.trainsplit
        else:
            self.splitname = args.testsplit

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass

    def __load_image__(self, path):
        img = cv.imread(path)
        img = cv.resize(img, (self.args.load_size, self.args.load_size))
        if img.shape[2] == 1:
            img = cv.cvtColor(img, cv.CV_GRAY2RGB)
        img = cv.normalize(img, None, alpha=0, beta=1, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)
        return img

    def __load_mask__(self, label):
        """Loading the facial mask corresponding to the label"""
        get_id = lambda id_name: self.splitname[10]
        path_to_masks = self.args.masks
        mask_sub_folder = "train_{}".format(get_id(self.splitname))
        mask_file_name = "mask_{}.png".format(label + 1)  # + 1 because of names id are label+1

        mask_path = os.path.join(path_to_masks, mask_sub_folder, mask_file_name)

        img = cv.imread(mask_path)

        img = cv.resize(img, (self.args.final_size, self.args.final_size))
        # grayscale
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        norm_img = cv.normalize(img, None, alpha=0, beta=1, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)

        norm_img = np.expand_dims(norm_img, axis=2)
        return norm_img


class CKP(DatasetBase):
    def __init__(self, args, train, transform=None):
        super(CKP, self).__init__(args, train=train)
        self.args = args
        self.transform = transform
        self.images_path = args.ckp_images
        self.path_csv = os.path.join(args.ckp_csvsplits, self.splitname)

        self.img_names = pd.read_csv(self.path_csv, index_col=False).values.tolist()
        self.img_names = [img_name[0] for img_name in self.img_names]
        self.img_paths, self.img_labels = self.__get_paths_and_labels__()

    def __get_paths_and_labels__(self):
        get_tag = lambda name: name[0:8]
        ds_tags = list(map(get_tag, self.img_names))

        ds_labels = pickle.load(open(self.args.labels, "rb"))

        get_tags = lambda name: (name[0:4], name[5:8])
        img_paths, labels = [], []
        for i in range(self.__len__()):
            imgname = self.img_names[i]
            tags = get_tags(imgname)
            imgtag = ds_tags[i]
            image_path = os.path.join(tags[0], tags[1], imgname)
            label = int(ds_labels[imgtag]) - 1  # -1 because CK+ has labels 1-7 -> convert to 0-6
            img_paths.append(image_path)
            labels.append(label)
        return img_paths, labels

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        label = self.img_labels[idx]
        # load image
        img = self.__load_image__(self.images_path + img_path)
        img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        img_gray = np.expand_dims(img_gray, axis=-1)  # (W;H;1)
        mask = self.__load_mask__(label)

        sample = {'image': img,
                  'image_gray': img_gray,
                  'label': label,
                  'mask': mask}

        if self.transform:
            sample = self.transform(sample)
        return sample


def get_ckp(args, batch_size=8, shuffle=True, num_workers=2):
    """Initialize ckp dataset and return train, test
    torch.utils.data.dataloader.DataLoader, already batched and shuffled."""
    transforms = tv.transforms.Compose([RandomCrop(args), ToTensor(), RandomFlip()])
    train_ds = CKP(train=True, args=args, transform=transforms)
    test_ds = CKP(train=True, args=args, transform=transforms)

    train_loader = DataLoader(dataset=train_ds,
                              batch_size=batch_size,
                              shuffle=shuffle,
                              num_workers=num_workers)

    test_loader = DataLoader(dataset=test_ds,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             num_workers=num_workers)

    return train_loader, test_loader


# #################################################################
#
# callable classes, can be used like a layer to stack in a model
#
# #################################################################
class RandomCrop(object):
    """Crop image of given sample to final_size"""

    def __init__(self, args):
        self.args = args

    def __call__(self, sample):
        image = sample['image']
        image_gray = sample['image_gray']
        mask = sample['mask']
        label = sample['label']

        h, w = image.shape[:2]
        new_h, new_w = self.args.final_size, self.args.final_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h, left: left + new_w]
        image_gray = image_gray[top: top + new_h, left: left + new_w]
        return {'image': image,
                'image_gray': image_gray,
                'label': label,
                'mask': mask}


class RandomFlip(object):
    """flip image and grayscaled image horizontally
    with a probability p"""

    def __call__(self, sample):
        image = sample['image']
        label = sample['label']
        image_gray = sample['image_gray']
        mask = sample['mask']

        image = tv.transforms.RandomHorizontalFlip(p=0.5)(image)
        image_gray = tv.transforms.RandomHorizontalFlip(p=0.5)(image_gray)

        return {'image': image,
                'image_gray': image_gray,
                'label': label,
                'mask': mask}


class GrayScale(object):
    """Convert a 3 channel image tensor into a 1 channel image tensor"""

    def __call__(self, sample):
        image = sample['image']
        image = tv.transforms.Grayscale(num_output_channels=1)(image)

        label = sample['label']
        image_gray = sample['image_gray']
        mask = sample['mask']
        sample = {'image': image,
                  'image_gray': image_gray,
                  'label': label,
                  'mask': mask}
        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors with channel first format"""

    def __call__(self, sample):
        image = sample['image']
        image_gray = sample['image_gray']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        image_gray = image_gray.transpose((2, 0, 1))

        label, mask = sample['label'], sample['mask']
        mask = mask.transpose((2, 0, 1))
        return {'image': to.from_numpy(image),
                'image_gray': to.from_numpy(image_gray),
                'label': to.tensor(label),
                'mask': to.from_numpy(mask)}
