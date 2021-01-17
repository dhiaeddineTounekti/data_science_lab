import os
class Config:
    WORKING_DIR = os.getcwd()
    
    # directory where all data will be put.
    DATASET_DIR = os.path.join(WORKING_DIR, "data")

    # the path of the original Kaggle dataset. Change it according to where you have put the dataset and what you have named the folder containing it.
    KAGGLE_3M_DATASET_DIR = os.path.join(DATASET_DIR, "lgg-mri-segmentation/kaggle_3m")

    GAN_TRAIN_DATASET = os.path.join(DATASET_DIR, "gan_dataset")
    GAN_TRAIN_MASK = os.path.join(GAN_TRAIN_DATASET, "A")
    GAN_TRAIN_MRI = os.path.join(GAN_TRAIN_DATASET, "B")
    GAN_COMBINED_DATASET = os.path.join(GAN_TRAIN_DATASET,"mask_mri")

    IMAGE_EXTENSION = ".jpg"

    # data split proportions
    TRAIN = 0.8
    VAL = 0.1
    TEST = 0.1

    # The path to the folder where to save the generated masks
    GAN_GENERATED_IMG = os.path.join(DATASET_DIR, "gan_generated_img")
    GAN_GENERATED_MASKS = os.path.join(GAN_GENERATED_IMG, "gan_generated_masks")
    GAN_GENERATED_MRI = os.path.join(GAN_GENERATED_IMG, "gan_generated_mri")

    # GAN model name
    GAN_MODEL_NAME = "mri_pix2pix"

