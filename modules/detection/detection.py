from tkinter import X
import torch
from torch import optim
import torch.nn.functional as F
from pytorch_lightning.core.lightning import LightningModule
from utils.utils import get_argparser_group
import wandb
wandb.login()
import numpy as np
import torch.nn as nn
import torchvision

class Detection(LightningModule):
    def __init__(self, hparams, model):
        super(Detection, self).__init__()
        self.save_hyperparameters(hparams)
        self.model = model
        self.example_input_array = torch.zeros(self.hparams.batch_size, self.hparams.input_channels,self.hparams.input_crop_size ,self.hparams.input_crop_size)
        self.loss = 0
        run = wandb.init()
    # 1: Forward step (forward hook), Lightning calls this inside the training loop
    def forward(self, x):
        model_output = self.model.forward(x) # the output is a list (batch* clip size) of dictionnaries 
        # made of tensors representing 'boxes','labels','scores'
        return model_output

    
    # 2: Optimizer (configure_optimizers hook)
    # see https://pytorch-lightning.readthedocs.io/en/latest/optimizers.html
    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.hparams.learning_rate, momentum=0.95, weight_decay=1e-5, nesterov=True)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=6, eta_min=0, verbose=True)
        return [optimizer], [scheduler]

    # 4: Training Loop (this is where the magic happens)(training_step hook)
    def training_step(self, train_batch, batch_idx):
        images, y_true = train_batch  # x is the image and y_true is the target dictionnary {'boxes','labels'}
        images = list(image for image in images)
        targets = []
        for i in range (len(images)):
            d = {}
            d['boxes'] = y_true['boxes'][i]
            d['labels'] = y_true['labels'][i]
            targets.append(d)

        #for model inference

        # model.eval()
        # predictions = model(x) #output as [{'boxes': tensor([], size=(0, 4), grad_fn=<StackBackward0>), 'labels': tensor([], dtype=torch.int64), 'scores': tensor([], grad_fn=<IndexBackward0>)}, {'boxes': tensor([], size=(0, 4), grad_fn=<StackBackward0>), 'labels': tensor([], dtype=torch.int64), 'scores': tensor([], grad_fn=<IndexBackward0>)}]

        # for model training 
        print('1')
        boxes, labels, scores = self.model(images, targets)
        print('2')
        loss_dict = {}
        # calculate loss from dict
        train_loss = sum(loss for loss in loss_dict.values()) # TO DO check how to compute loss from box and class
        # in DP mode (default) make sure if result is scalar, there's another dim in the beginning
        # if self.trainer.use_dp or self.trainer.use_ddp2:
        #     train_loss = train_loss.unsqueeze(0)
            
        return {'loss': train_loss, 'y_true': y_true}
    #  F.log_softmax(torch.reshape(logits,(self.hparams.batch_size,18,self.hparams.input_clip_length)), dim=1)
    # If using metrics in data parallel mode (dp), the metric update/logging should be done in the training_step_end
    # This is due to metric states else being destroyed after each forward pass, leading to wrong accumulation.
    def training_step_end(self, train_step_output):

        self.log('Train Loss', train_step_output['loss'], on_epoch=False, on_step=True)
        wandb.log({'Train_loss': train_step_output['loss']})
        return train_step_output

    # 5 Validation Loop
    def validation_step(self, val_batch, batch_idx):
        images, y_true = val_batch  # x is the image and y_true is the target dictionnary {'boxes','labels'}
        images = list(image for image in images)
        targets = []
        for i in range (len(images)):
            d = {}
            d['boxes'] = y_true['boxes'][i]
            d['labels'] = y_true['labels'][i]
            targets.append(d)

        #for model inference

        # model.eval()
        # predictions = model(x) #output as [{'boxes': tensor([], size=(0, 4), grad_fn=<StackBackward0>), 'labels': tensor([], dtype=torch.int64), 'scores': tensor([], grad_fn=<IndexBackward0>)}, {'boxes': tensor([], size=(0, 4), grad_fn=<StackBackward0>), 'labels': tensor([], dtype=torch.int64), 'scores': tensor([], grad_fn=<IndexBackward0>)}]

        # for model training 

        print('val 1')
        boxes, labels, scores = self.model(images, targets)
        print('val 2')        # calculate loss from dict
        loss_dict={}
        val_loss = sum(loss for loss in loss_dict.values()) # TO DO check how to compute loss from box and class
        # in DP mode (default) make sure if result is scalar, there's another dim in the beginning
        if self.trainer.use_dp or self.trainer.use_ddp2:
            train_loss = train_loss.unsqueeze(0)
        return {'loss': val_loss, 'y_true': y_true}

    # If using metrics in data parallel mode (dp), the metric update/logging should be done in the validation_step_end
    # This is due to metric states else being destroyed after each forward pass, leading to wrong accumulation.
    def validation_step_end(self, val_step_output):
        self.log('Validation Loss', val_step_output['loss'], on_epoch=False, on_step=True)

        wandb.log({'Validation_loss': val_step_output['loss']})
        return val_step_output
        
    # no test loop given the MOMA dataset...
    # 6 Test Loop
    def test_step(self, test_batch, batch_idx):
        x, y_true = test_batch
        y_pred = self(x)
        # print('y_pred shape is',y_pred.shape,'y_true shape is',y_true.shape)
        # print('y_pred looks like this',y_pred, 'and y_true looks like this',y_true)
        loss = self.loss_function(y_pred, y_true)
        return {'test_loss': loss, 'y_pred': y_pred, 'y_true': y_true}

    def test_step_end(self, test_step_output):
        # self.log('test_loss', test_step_output['test_loss'], on_step=True, on_epoch=True)
        return test_step_output
    
    

    @staticmethod
    def add_module_specific_args(parser):
        specific_args = get_argparser_group(title='Model options', parser=parser)
        return parser
