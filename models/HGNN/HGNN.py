from torch import nn
from models import HGNN_conv
import torch.nn.functional as F
import pytorch_lightning as pl

class HGNN(pl.LightningModule):
    def __init__(self, hparams):
        self.in_ch=hparams['in_ch']
        self.in_ch=hparams['n_hid']
        self.in_ch=hparams['n_class']
        self.dropout=hparams['dropout']
        self.hgc1 = HGNN_conv(self.in_ch, self.n_hid)
        self.hgc2 = HGNN_conv(self.n_hid, self.n_class)

        self.actor_pool = nn.AvgPool3d(kernel_size=(16, 10, 10), stride=1, padding=0)
        self.actor_conv = nn.Conv3d(432, 2048, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
        self.actor_act = nn.ReLU()
        self.actor_dropout = nn.Dropout(p=0.5, inplace=False)

        self.frame_pool = nn.AvgPool3d(kernel_size=(16, 10, 10), stride=1, padding=0)
        self.frame_conv = nn.Conv3d(432, 2048, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
        self.frame_act = nn.ReLU()
        self.frame_dropout = nn.Dropout(p=0.5, inplace=False)

        self.video_pool = nn.AvgPool3d(kernel_size=(16, 10, 10), stride=1, padding=0)
        self.video_conv = nn.Conv3d(432, 2048, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
        self.video_act = nn.ReLU()
        self.video_dropout = nn.Dropout(p=0.5, inplace=False)


    def forward(self, x, G):
        x = F.relu(self.hgc1(x, G))
        x = F.dropout(x, self.dropout)
        x = self.hgc2(x, G)
        # x's format is n_batch * numbers of nodes * features size(same size input feature)
        
        # We have 3 readout level: actor-centric, frame-level, video level
        # understood as different pooling
        actor = self.actor_pool(x)
        actor = self.actor_conv(actor)
        actor = self.actor_act(actor)
        actor = self.actor_dropout(actor)

        frame = self.frame_pool(x)
        frame = self.frame_conv(frame)
        frame = self.frame_act(frame)
        frame = self.frame_dropout(frame)
        
        video = self.video_pool(x)
        video = self.video_conv(video)
        video = self.video_act(video)
        video = self.video_dropout(video)

        return actor, frame, video