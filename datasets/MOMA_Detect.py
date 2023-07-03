from cmath import nan
from inspect import EndOfBlock
# from pytorchvideo.transforms import (
#     RandAugment, )
from utils.utils import get_argparser_group
import torchvision.transforms as transforms
import os
import random
import torch
from pathlib import Path
from torch.functional import split
from torch.utils.data import Dataset
import pytorch_lightning as pl
import numpy as np
import pandas as pd
from torchvision.transforms.transforms import Normalize, Compose, Resize, ToTensor, RandomAffine, RandomHorizontalFlip, RandomCrop, RandomRotation
from torch.utils.data import DataLoader
import cv2
import numpy as np
import json
from PIL import Image

def one_hot(a, num_classes):
    encoding=np.zeros(num_classes)
    for index in a:
        encoding[index]=1
    return encoding.tolist()

class MOMA_Detect(Dataset):
    def __init__(self, hparams, split, add_transforms=None):
        super().__init__()
        self.hparams = hparams
        self.train_path=self.hparams.train_path
        self.val_path= self.hparams.val_path
        self.split=split
        if self.split == 'train':
            f= open(self.train_path)
            self.dict =json.load(f)
            f.close()
        elif self.split=='val':
            f= open(self.val_path)
            self.dict =json.load(f)
            f.close()        
        self.add_transforms = add_transforms
        self.batch_size = self.hparams.batch_size
        self.transforms = Compose([
            Resize((hparams.input_crop_size, hparams.input_crop_size)),
            ToTensor(),
        ])

    def __len__(self):
        return len(self.dict['index'])     

    def __getitem__(self,index):
        path=self.dict['path'][index]
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        # image=self.transforms(Image.fromarray(np.uint8(image)))
        image /= 255.0
        labels=self.dict['classes'][index]
        target={}
        target['boxes']=self.dict['bboxes'][index]
        target['labels']= torch.squeeze(torch.as_tensor(labels, dtype=torch.int64))
        if self.transforms:
            sample = {
                'image': image,
                'bboxes': target['boxes'],
                'labels': target['labels']
            }

        target['boxes'] = torch.as_tensor(sample['bboxes'])
        return image,target
    
    
class MOMA_Detection(pl.LightningDataModule):
    """
    This class defines the MOMA dataset.
    split : .data['train'], .data['val'] accordingly to the dataset split

    Arguments:
        hparams: config from pytorch lightning
    """
    def __init__(self, hparams):
        super().__init__()
        print("dataset initialization ...")
        self.save_hyperparameters(hparams)
        self.data_root = Path(hparams.data_root)
        self.out_features = self.hparams.out_features
        self.input_channels = self.hparams.input_channels
        self.batch_size = self.hparams.batch_size
        self.transformations = self.__get_transformations()
        self.data = {}
        self.train_path=self.hparams.train_path
        self.val_path= self.hparams.val_path
        val_transforms = Compose([
        Resize((hparams.input_crop_size, hparams.input_crop_size)),
        ToTensor(),
        ])
        train_transforms = Compose(
        [val_transforms,
        # RandAugment(magnitude=5, num_layers=3)
        ])
        print('done')
        

    def __dataloader(self, split=None):

        # don't shuffle for validation and testing, only shuffle for training
        shuffle = split == 'train'
        train_sampler = None
        # when using multi-node ddp, add data sampler
        # if self.use_ddp:
        #     train_sampler = DistributedSampler(dataset)
        if split == 'train':
            dataset=MOMA_Detect(hparams=self.hparams,split='train')
        else:
            dataset=MOMA_Detect(hparams=self.hparams,split='val')
     
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.hparams.batch_size,
            shuffle=shuffle,
            sampler=train_sampler,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
        )
        return dataloader

    def train_dataloader(self):
        dataloader = self.__dataloader(split='train')
        return dataloader
        # if you want to skip the __dataloader function:
        # return DataLoader(self.data['train'], pin_memory=True, num_workers=self.num_workers,
        #                   batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        dataloader = self.__dataloader(split='val')
        return dataloader
        # if you want to skip the __dataloader function:
        # return DataLoader(self.data['val'], pin_memory=True, num_workers=self.num_workers,
        #                   batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        dataloader = self.__dataloader(split='test')
        return dataloader
        # if you want to skip the __dataloader function:
        # return DataLoader(self.data['test'], pin_memory=True, num_workers=self.num_workers,
                        #   batch_size=self.batch_size, shuffle=False)

    def set_num_channels(self, num_channels):
        """
        Set the number of channels of the input
        :param num_channels:
        :return: input channels as int
        """
        self.num_channels = num_channels

    def __get_transformations(self):
        """
        set data transformations including:
         - augmentation
         - normalization
         - conversion to tensor

        returns a list with transformations for 'train', 'val' and 'test'
        """
        # identity in case of only one channel
        expand_channels = transforms.Lambda(lambda img: img)
        # if more than 1 channel expand image to number of channels
        if self.input_channels > 1:
            print(
                f'{self.name}: expanding input image to {self.input_channels} channels'
            )
            expand_channels = transforms.Lambda(
                lambda img: img.view(1, self.input_height, self.input_width).expand(self.input_channels, -1, -1)
            )
        data_transformations = {}
        for split in ['train', 'val', 'test']:
            data_transformations[split] = transforms.Compose(
                [expand_channels,transforms.ToTensor()]
            )
        return data_transformations

    @staticmethod
    def add_dataset_specific_args(parser):
        """
        Parameters you define here will be available to your model through self.hparams
        :param parser:
        :param root_dir:
        :return:
        """
        mnist_specific_args = get_argparser_group(title='Dataset options', parser=parser)
        mnist_specific_args.add_argument('--dataset_disable_train_transform', action='store_true')

        mnist_specific_args.add_argument('--input_height', default=256, type=int)
        mnist_specific_args.add_argument('--input_width', default=256, type=int)
        mnist_specific_args.add_argument('--out_features', default=17, type=int)
        return parser
    
    
    
    
    
    