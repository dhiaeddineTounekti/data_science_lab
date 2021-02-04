import json
import os

import albumentations as A
import cv2
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset

import config


class CustomDataset(Dataset):
    """Brain MRI dataset"""

    def __init__(self, mask_rcnn: bool = False, train: bool = False, mean=None, std=None, real_only: bool = True):
        """
        initiate the Dataset
        :param train:
        :param mask_rcnn: parameter to know if we are using mask rcnn or unet because they train differently
        """
        self.conf = config.Config()
        # Open json file.
        file = open(self.conf.FILE_PATH, 'r')
        # Data dict.
        data = json.loads(file.read())
        # Get key, value pairs of real data and generated data.
        (_, self.real_data), (_, self.generated_data) = data.items()

        self.is_mask_rcnn = mask_rcnn
        self.train = train

        # Make a list of composition to use
        self.transform = T.Compose([
            T.ToTensor()
        ])

        self.real_data_only = real_only

        self.mean = mean
        self.std = std

        self.train_transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
        ])

    def __len__(self):
        if self.real_data_only:
            return self.real_data_size()
        return self.real_data_size() + self.gan_generated_data_size()

    def real_data_size(self):
        """
        Returns the length of the real dataset
        :return:
        """
        return len(self.real_data["mask_boxes_and_classes"])

    def gan_generated_data_size(self):
        """
        Returns the length of the gan_generated dataset
        :return:
        """
        return len(self.generated_data["mask_boxes_and_classes"])

    def __getitem__(self, idx):
        """
        get an item using index
        You need to be careful here because we are going to merge two data types (GAN_generated and real data)
        :param idx: index of the element.
        :return:
        """
        # If the index is more than the real data length then it must be one of the generated data
        # You should make sure that when you concat both lists to create the dataset you have to concat them like this:
        # [Real data set, generated data set]
        is_real_data = True
        if idx >= self.real_data_size():
            is_real_data = False
            new_index = idx - self.real_data_size()
            image_path = os.path.join(self.conf.GAN_GENERATED_MRI,
                                      self.generated_data["mask_boxes_and_classes"][new_index]['name'])
            mask_path = os.path.join(self.conf.GAN_GENERATED_MASKS,
                                     self.generated_data["mask_boxes_and_classes"][new_index]['name'])
            box = self.generated_data["mask_boxes_and_classes"][new_index]['box']

        else:
            image_path = os.path.join(self.conf.REAL_TRAIN_MRI, self.real_data["mask_boxes_and_classes"][idx]['name'])
            mask_path = os.path.join(self.conf.REAL_TRAIN_MASK, self.real_data["mask_boxes_and_classes"][idx]['name'])
            box = self.real_data["mask_boxes_and_classes"][idx]['box']

        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if is_real_data and self.train:
            # Use augmentation only on real data and during training.
            transformed = self.train_transform(image=img, mask=mask)
            img = transformed['image']
            mask = transformed['mask']

        # Make the image in the interval [0-1] and convert images to tensors
        img = self.transform(img)
        mask = self.transform(mask)

        if self.mean is not None and self.std is not None:
            img = T.Normalize(mean=self.mean, std=self.std)(img)

        target = {
            'masks': mask,
            'boxes': torch.FloatTensor([box]) if box != [] else torch.FloatTensor([[0, 0, 1, 1]]),
            'labels': torch.tensor([1], dtype=torch.int64) if box != [] else torch.tensor([0], dtype=torch.int64)
        }

        if self.is_mask_rcnn and self.train:
            # 1 here denotes the class which is "cancer" if the box exists.
            # else if the box does not exist it will return 0 which denotes "background".
            # This is only relevant for mask rcnn
            # Put a dummy box if there are no elements
            return img, target

        return img, mask
