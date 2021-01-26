import os
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optimizer
from torch.nn import DataParallel
from torch.utils.data import SubsetRandomSampler, DataLoader

import src.config as config
import src.data_loaders.dataset as dataset
from src.models.mask_rcnn import MaskRCNN
from src.models.u_net import UNet

conf = config.Config()
best_eval_loss = 10


def get_dataset_split(real_dataset_size: int, gan_generated_dataset_size: int, augmented: bool = False) -> tuple:
    """
    Split the data into train, validation and test.
    make sure that the gan generated data is not included in the validation and test dataset.
    :param augmented: controls if the gan generated data are included or not.
    :param gan_generated_dataset_size: size of the gan generated dataset.
    :param real_dataset_size: real dataset size.
    :return:
    """
    real_index = list(range(real_dataset_size))
    gan_data_index = list(range(gan_generated_dataset_size))
    # Shuffle list indexes
    random.shuffle(real_index)
    random.shuffle(gan_data_index)

    val_index_start = int(conf.TRAIN * real_dataset_size)
    test_index_start = int(conf.VAL * real_dataset_size) + val_index_start

    # Assign to each split the corresponding indexes
    train_indexes = real_index[:val_index_start] + gan_data_index if augmented else real_index[:val_index_start]
    val_indexes = real_index[val_index_start: test_index_start]
    test_indexes = real_index[test_index_start:]

    return SubsetRandomSampler(train_indexes), SubsetRandomSampler(val_indexes), SubsetRandomSampler(test_indexes)


def create_model_dict() -> dict:
    return {
        'mask_rcnn': {
            'model': MaskRCNN(),
            'data': {
                'real_data': {
                    'train': DataLoader(dataset=mask_rcnn_dataset_train, sampler=train_sampler,
                                        batch_size=conf.BATCH_SIZE,
                                        num_workers=conf.NUM_WORKERS),
                    'val': DataLoader(dataset=mask_rcnn_dataset_train, sampler=val_sampler, batch_size=conf.BATCH_SIZE,
                                      num_workers=conf.NUM_WORKERS),
                    'test': DataLoader(dataset=mask_rcnn_dataset_train, sampler=test_sampler,
                                       batch_size=conf.BATCH_SIZE,
                                       num_workers=conf.NUM_WORKERS)
                },
                'gan_data': {
                    'train': DataLoader(dataset=mask_rcnn_dataset_train, sampler=train_sampler_augmented,
                                        batch_size=conf.BATCH_SIZE, num_workers=conf.NUM_WORKERS),
                    'val': DataLoader(dataset=mask_rcnn_dataset_train, sampler=val_sampler_augmented,
                                      batch_size=conf.BATCH_SIZE, num_workers=conf.NUM_WORKERS),
                    'test': DataLoader(dataset=mask_rcnn_dataset_train, sampler=test_sampler_augmented,
                                       batch_size=conf.BATCH_SIZE, num_workers=conf.NUM_WORKERS)
                }
            }
        },
        'unet': {
            'model': UNet(),
            'data': {
                'real_data': {
                    'train': DataLoader(dataset=unet_dataset_train, sampler=train_sampler, batch_size=conf.BATCH_SIZE,
                                        num_workers=4),
                    'val': DataLoader(dataset=unet_dataset_train, sampler=val_sampler, batch_size=conf.BATCH_SIZE,
                                      num_workers=4),
                    'test': DataLoader(dataset=unet_dataset_train, sampler=test_sampler, batch_size=conf.BATCH_SIZE,
                                       num_workers=4)
                },
                'gan_data': {
                    'train': DataLoader(dataset=unet_dataset_train, sampler=train_sampler_augmented,
                                        batch_size=conf.BATCH_SIZE, num_workers=4),
                    'val': DataLoader(dataset=unet_dataset_train, sampler=val_sampler_augmented,
                                      batch_size=conf.BATCH_SIZE, num_workers=4),
                    'test': DataLoader(dataset=unet_dataset_train, sampler=test_sampler_augmented,
                                       batch_size=conf.BATCH_SIZE, num_workers=4)
                }
            }
        }
    }


def invert_dict_of_tensors(target):
    """
    Change dict{ 'a': Tensor, 'b': Tensor ...} to list[ Dict {'a': Tensor ...}]
    :param target: dictionary to change
    :return:
    """
    result = []
    for idx in range(len(target['boxes'])):
        dictionary = {
            'masks': target['masks'][idx],
            'boxes': target['boxes'][idx],
            'labels': target['labels'][idx]
        }
        result.append(dictionary)

    return result


def train(target_model: nn.Module, data_loader: DataLoader, opt: optimizer.Optimizer, epoch: int,
          model_name: str, data_type: str) -> None:
    """
    Trains the model and logs the training result in a file.
    :param data_type: the name of the data loader wither it is real or gan augmented
    :param target_model: the model to train
    :param opt: the optimizer
    :param epoch: the current number of epochs
    :param model_name: the model name
    :return:
    """
    # TODO : split this function into train and validate and use model.eval mode in validation function.
    target_model.train()

    # Initiate the loss
    loss_sum = 0.0

    # Open Log file
    file_name = os.path.join(conf.CHECKPOINT_FOLDER, f'{model_name}_{data_type}_training_loss.log')
    log_file = open(file_name, 'a+')

    start_time_epoch = time.time()
    for batch_idx, (data, target) in enumerate(data_loader):
        start_time = time.time()
        opt.zero_grad()

        # Invert the dict of tensors you get from the MaskRCNN
        if type(target_model.module) == MaskRCNN:
            inverted_target_output = invert_dict_of_tensors(target)
            output = target_model(data, inverted_target_output)
            loss = torch.stack([loss for key, loss in output.items()]).sum()

        else:
            output = target_model(data)
            loss = f.binary_cross_entropy_with_logits(output, target)

        loss_sum += float(loss)

        loss.backward()
        opt.step()

        if batch_idx and batch_idx % 10 == 0:
            message = f'Epoch: {epoch} | Iteration number: [{batch_idx}/{len(data_loader)}] {batch_idx // len(data_loader) * 100}% | Training loss: {loss_sum / batch_idx}\n'
            log_file.writelines([message])

    message = f'End of epoch: {epoch} | Train Loss: {loss_sum / len(data_loader)} | Training Time: {int(time.time() - start_time_epoch)}'
    log_file.write(f'\n {message} \n')
    log_file.close()


def validate(target_model: nn.Module, data_loader: DataLoader, epoch: int,
             model_name: str, data_type: str) -> None:
    """
    validate the model.
    :param model_name:
    :param data_loader:
    :param epoch:
    :param data_type:
    :param target_model:
    :return:
    """
    target_model.eval()
    global best_eval_loss

# Initiate the loss
    loss_sum = 0.0

    # Open Log file
    file_name = os.path.join(conf.CHECKPOINT_FOLDER, f'{model_name}_{data_type}_training_loss.log')
    log_file = open(file_name, 'a+')

    start_time_epoch = time.time()

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(data_loader):
            if type(target_model.module) == MaskRCNN:
                output = target_model(data)
                # TODO : find a way to calculate the loss. Calculate only mask loss.
            else:
                output = target_model(data)
                loss = f.binary_cross_entropy_with_logits(output, target)

            loss_sum += float(loss)

    message = f'End of epoch: {epoch} | Eval Loss: {loss_sum / len(data_loader)} | Evaluating Time: {int(time.time() - start_time_epoch)}'
    log_file.write(f'\n {message} \n')
    log_file.close()

    if loss_sum / len(data_loader) < best_eval_loss:
        best_eval_loss = loss_sum / len(data_loader)
        torch.save(target_model, f'{model_name}_{data_type}_ep_{epoch}_best_model.pth')


if __name__ == "__main__":
    # Check if checkpoint folder are there
    if not os.path.isdir(conf.CHECKPOINT_FOLDER):
        os.mkdir(conf.CHECKPOINT_FOLDER)

    unet_dataset_train = dataset.CustomDataset(mask_rcnn=False, train=True)
    mask_rcnn_dataset_train = dataset.CustomDataset(mask_rcnn=True, train=True)
    test_dataset = dataset.CustomDataset(train=False)

    # Make train sampler for gan augmented data and real_data
    train_sampler, val_sampler, test_sampler = get_dataset_split(unet_dataset_train.real_data_size(),
                                                                 unet_dataset_train.gan_generated_data_size())
    train_sampler_augmented, val_sampler_augmented, test_sampler_augmented = get_dataset_split(
        unet_dataset_train.real_data_size(),
        unet_dataset_train.gan_generated_data_size(), augmented=True)

    # Prepare data loaders for different models
    models_dict = create_model_dict()

    for model_name, value in models_dict.items():
        # Get the model to work in a multi-gpu environment
        model = DataParallel(value['model'])
        # Prepare optimizer
        opt = optimizer.Adam(lr=0.0001, params=model.parameters())

        # Iterate through both kind of datasets:
        for data_type, dataloader_dict in value['data'].items():
            global best_eval_loss
            best_eval_loss = 100
            for epoch in range(1, conf.EPOCHS_SEG + 1):
                train(target_model=model, data_loader=dataloader_dict['train'], opt=opt, epoch=epoch,
                      model_name=model_name, data_type=data_type)
                validate(target_model=model, data_loader=dataloader_dict['val'], epoch=epoch, model_name=model_name,
                         data_type=data_type)
