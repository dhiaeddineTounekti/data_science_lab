import json
import os

import cv2
from torch.utils.data import Dataset
import src.config as config


class CustomDataset(Dataset):
    """Brain MRI dataset"""

    def __init__(self, transform=None, mask_rcnn=False):
        """
        initiate the Dataset
        :param transform: transformation to apply to each
        :param mask_rcnn:
        """
        self.conf = config.Config()
        # Open json file.
        file = open(self.conf.FILE_PATH, 'r')
        # Data dict.
        data = json.loads(file.read())
        self.real_data, self.generated_data = data.items()

        self.is_mask_rcnn = mask_rcnn
        self.transform = transform

    def __len__(self):
        return len(self.real_data["mask_boxes_and_classes"]) + len(self.generated_data["mask_boxes_and_classes"])

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
        if idx > len(self.real_data):
            new_index = idx - len(self.real_data)
            image_path = os.path.join(self.conf.GAN_GENERATED_MRI, self.generated_data[new_index]['name'])
            mask_path = os.path.join(self.conf.GAN_GENERATED_MASKS, self.generated_data[new_index]['name'])
            box = self.generated_data[new_index]['box']

        else:
            image_path = os.path.join(self.conf.REAL_TRAIN_MRI, self.real_data[idx]['name'])
            mask_path = os.path.join(self.conf.REAL_TRAIN_MASK, self.real_data[idx]['name'])
            box = self.real_data[idx]['box']

        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)

        if self.transform:
            img, mask = self.transform(image=img, segmentation_maps=mask)

        if self.is_mask_rcnn:
            # 1 here denotes the class which is clearly only cancer if the box exists.
            # else if the box does not exist it will return 0 which denotes background.
            return img, mask, box, 1 if box != [] else 0

        return img, mask
