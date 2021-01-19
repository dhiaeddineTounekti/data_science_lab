import config
import numpy as np
import cv2
import os
import shutil


def generate_boxes():
    """
    Generate boxes coordinates for each mask
    :return:
    """
    # First generate masks for the generated shapes
    conf = config.Config()
    gan_mask_dict = []
    real_mask_dict = []
    # Get the gan mask dict
    for image_name in os.listdir(conf.GAN_GENERATED_MASKS):
        image_path = os.path.join(conf.GAN_GENERATED_MASKS, image_name)
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        thresh = 100
        ret, thresh_img = cv2.threshold(img_grey, thresh, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        bounding_rec = cv2.boundingRect(contours[0])
        gan_mask_dict.append({
            'name': image_name,
            'box': bounding_rec
        })

    for image_name in os.listdir(conf.GAN_GENERATED_MASKS):
        image_path = os.path.join(conf.GAN_GENERATED_MASKS, image_name)
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        thresh = 100
        ret, thresh_img = cv2.threshold(img_grey, thresh, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        bounding_rec = cv2.boundingRect(contours[0])
        gan_mask_dict.append({
            'name': image_name,
            'box': bounding_rec
        })


def group_data():
    """
    Move real masks and real mri to corresponding directories
    :return:
    """
    conf = config.Config()
    if not os.path.isdir(conf.REAL_DATA_DIR):
        os.mkdir(conf.REAL_DATA_DIR)

    if not os.path.isdir(conf.REAL_TRAIN_MRI):
        os.mkdir(conf.REAL_TRAIN_MRI)

    if not os.path.isdir(conf.REAL_TRAIN_MASK):
        os.mkdir(conf.REAL_TRAIN_MASK)

    idx = 0
    for patient_folder in os.listdir(conf.KAGGLE_3M_DATASET_DIR):
        if os.path.isdir(patient_folder):
            for image in os.listdir(patient_folder):
                # Get a mask
                if image.find("_mask") != -1:
                    # Copy real mask
                    current_mask_path = os.path.join(conf.KAGGLE_3M_DATASET_DIR, patient_folder, image)
                    target_mask_path = os.path.join(conf.REAL_TRAIN_MASK, str(idx) + ".png")
                    shutil.copyfile(current_mask_path, target_mask_path)

                    # Copy image
                    current_image_path = os.path.join(conf.KAGGLE_3M_DATASET_DIR, patient_folder,
                                                      image.replace("_mask", ""))
                    target_image_path = os.path.join(conf.REAL_TRAIN_MRI, str(idx) + ".png")
                    shutil.copyfile(current_image_path, target_image_path)

                    # Advance counter
                    idx += 1
