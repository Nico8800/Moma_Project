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
from tqdm import tqdm
from PIL import Image
from torch.utils.data import DataLoader
from tabulate import tabulate
import matplotlib.pyplot as plt
import json
from tqdm import tqdm
from random import randint

from torchvision.utils import make_grid, save_image
import cv2
import numpy as np
from ast import literal_eval

def one_hot(a, num_classes):
    encoding=np.zeros(num_classes)
    for index in a:
        encoding[index]=1
    return encoding.tolist()



class MOMADataset(Dataset):
    def __init__(self, hparams, split, add_transforms=None):
        super().__init__()
        self.hparams = hparams
        self.df_train_path=self.hparams.df_train_path
        self.df_val_path= self.hparams.df_val_path
        self.input_clip_length=hparams.input_clip_length
        self.split=split
        if self.split == 'train':
            self.df= pd.read_csv(self.df_train_path)
        elif self.split=='val':
            self.df=pd.read_csv(self.df_val_path)
        self.add_transforms = add_transforms
        self.out_features = self.hparams.out_features
        self.input_channels = self.hparams.input_channels
        self.batch_size = self.hparams.batch_size
        self.transforms = Compose([
            Resize((hparams.input_crop_size, hparams.input_crop_size)),
            ToTensor(),
        ])
    def __len__(self):
        return len(self.df)     
    # TO DO think of quicker way to extract frames 
    def __getitem__(self,index):
        raw_vid_id=self.df.at[index, "raw_video_id"]
        trim_video_id=self.df.at[index, "trim_video_id"]
        start=int(literal_eval(self.df.at[index,"frame_ids"])[0])
        end=int(literal_eval(self.df.at[index,"frame_ids"])[-1])
        video=[]
        activity_label=[self.df.at[index, "activity_labels"]]
        subactivity_label=[self.df.at[index, "sub_activity_labels"]]
        aactivity_label=[self.df.at[index, "atomical_action_label"]]
        item_path = self.df.at[index, "path"]
        # print("height is {} , and width is {}".format(frameHeight,frameWidth))
        # print( "end start is of lenght {}".format(end-start))
        # alternative with negative end 
        labels=[]
        activity_labels=[]
        sub_labels=[]
        aact_labels=[]
        empty_frames=0
        for counter,frame in enumerate(range(start,end+1)):
            # print("counter is {} and frame is {}".format(counter,frame))
            if frame >=0:
                path_to_image=os.path.join('/home/guests/nicolas_gossard/moma_data/frame_data',str(trim_video_id),str(frame)+'.jpg')
                image=cv2.imread(path_to_image)
                video.append(self.transforms(Image.fromarray(np.uint8(image))))
                # print(aactivity_label)
                aact_labels.append(literal_eval(aactivity_label[0])[counter-empty_frames])
            if frame<0:
                empty_frames+=1
                video.append(self.transforms(Image.fromarray(np.uint8(np.zeros((self.hparams.input_crop_size,self.hparams.input_crop_size,3))))))
                aact_labels.append([0])
        activity_labels.append(int(activity_label[0]))
        sub_labels.append(subactivity_label[0])
        stacked_encoding=[np.zeros((52)) for _ in range(16)]
        for index,encoding_frame in enumerate(aact_labels):
            encoding=one_hot(encoding_frame,52)
            stacked_encoding[index]=encoding
        labels.append(torch.tensor(activity_label[0]))
        labels.append(torch.tensor(sub_labels[0]))
        labels.append(torch.tensor(stacked_encoding))
        video = torch.stack(video)
        if self.add_transforms:
            video = self.add_transforms(video)

        assert torch.is_tensor(video)
        assert isinstance(labels, list)
        # if (any(torch.flatten(video)<0) or any(torch.flatten(video)>255)):
        # print("What is going on the video doesn t have the correct values",any(torch.flatten(video)<0),any(torch.flatten(video)>256))
        #save_image(video, "/home/tobicz/example_out/example.png")
        # print(video.shape,labels.shape)
        return video,labels
    
    
class MOMA(pl.LightningDataModule):
    """
    This class defines the MOMA dataset.
    split : .data['train'], .data['val'] accordingly to the dataset split

    Arguments:
        hparams: config from pytorch lightning
    """
    def __init__(self, hparams):
        super().__init__()
        print("dataset initialization ...")
        self.hparams = hparams
        self.data_root = Path(hparams.data_root)
        self.out_features = self.hparams.out_features
        self.input_channels = self.hparams.input_channels
        self.batch_size = self.hparams.batch_size
        self.transformations = self.__get_transformations()
        self.data = {}
        self.df_train_path=self.hparams.df_train_path
        self.df_val_path= self.hparams.df_val_path
        val_transforms = Compose([
        Resize((hparams.input_crop_size, hparams.input_crop_size)),
        ToTensor(),
        ])
        train_transforms = Compose(
        [val_transforms,
        # RandAugment(magnitude=5, num_layers=3)
        ])
        print('done')
        
    def prepare_data(self):
        # download
        self.data_root='/mnt/ssd2/nicolasg/moma_data/video_data/trim_sample_videos'
        train_df=pd.read_csv(self.df_train_path)
        val_df=pd.read_csv(self.df_val_path)
        self.data['vid_path']=train_df['path']
        self.data['activity_label']=train_df['activity_labels']
        self.data['vid_path']=train_df['path']
        self.data['activity_label']=train_df['activity_labels']
    def setup(self, stage=None):
        # split the dataset into train, val, test
        # self.data.update(
        #     split_dataset(self.data['all'], test_size=self.test_size, val_size=self.val_size)
        # )
        # apply transformation if any
        # for key in self.transformations.keys():
        #     self.data[key] = TransformDataset(
        #         dataset=self.data[key], transform=self.transformations[key]
        #     )
        pass
    def __dataloader(self, split=None):
        if split == 'train':
            train_csv_path=self.df_train_path
            df=pd.read_csv(train_csv_path)
        if split == 'val':
            val_csv_path=self.df_val_path
            df= df=pd.read_csv(val_csv_path)
        # don't shuffle for validation and testing, only shuffle for training
        shuffle = split == 'train'
        train_sampler = None
        # when using multi-node ddp, add data sampler
        # if self.use_ddp:
        #     train_sampler = DistributedSampler(dataset)
        if split == 'train':
            dataset=MOMADataset(hparams=self.hparams,split='train')
        else:
            dataset=MOMADataset(hparams=self.hparams,split='val')
     
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
    
    
    
    
    
    