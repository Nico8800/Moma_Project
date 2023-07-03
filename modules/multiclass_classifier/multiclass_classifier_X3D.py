"""This is the module to the multiclass classification example.
It was run with the MNIST dataset and works with both MNIST models (mnist_ex1 and mnist_ex2 in the model folder)."""
import torch
import numpy as np
from torch import optim
from pytorch_lightning.metrics import Precision
from pytorch_lightning.core.lightning import LightningModule
from utils.utils import get_argparser_group
import wandb
wandb.login()
import torch.nn as nn
from torchmetrics import AveragePrecision
from utils.metrics import map

class MultiClassClassification(LightningModule):
    def __init__(self, hparams, model):
        super(MultiClassClassification, self).__init__()
        self.hparams = hparams
        self.model = model
        self.example_input_array = torch.zeros(self.hparams.batch_size,self.hparams.input_clip_length, self.hparams.input_channels,self.hparams.input_crop_size ,self.hparams.input_crop_size)
        self.aactPrecision = Precision(num_classes=52, average='macro')
        self.subactPrecision = Precision(num_classes=68, average='macro')
        self.actPrecision = Precision(num_classes=17, average='macro')
        self.softmax=nn.Softmax(dim=1)
        self.sigmoid=nn.Sigmoid()
        self.BCEloss=nn.BCEWithLogitsLoss() # TO DO mask loss for the black frames 
        self.CEloss=nn.CrossEntropyLoss()
        self.mAP=map()
        # self.AP_aact=AveragePrecision(num_classses= 53,pos_label=1)
        # self.AP_subact=AveragePrecision(num_classses=68)
        # self.AP_act=AveragePrecision(num_classes= 17)
        run = wandb.init()
    # we want to reformat the true label of atomic actions we got batches for every element of the onehot encoding
    # def reformat_true(self,y_true):
    #     activity=y_true[0]
    #     subact=y_true[1]
    #     to_fix=y_true[2:]
    #     fixed=[]
    #     for batch in range(len(to_fix[-1][-1])):
    #         temp_fixed=[]
    #         for frame in range(self.hparams.input_clip_length):
    #             temp_list=[]
    #             for i in range(len(to_fix[-1])):
    #                 temp_list.append(int(to_fix[frame][i][batch].item()))
    #             temp_fixed.append(temp_list)
    #         fixed.append(temp_fixed)
    #     return [activity,subact,fixed]
    # 1: Forward step (forward hook), Lightning calls this inside the training loop
    def forward(self, x):
        act,subact,aact = self.model.forward(x)
        return [act,subact,aact]

    
    # 2: Optimizer (configure_optimizers hook)
    # see https://pytorch-lightning.readthedocs.io/en/latest/optimizers.html
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        # where T_max is the maximum number of iterations
        return [optimizer], [scheduler]
    # def aact_loss(self,y_pred,y_true):
    #     # sftmx=nn.Softmax(dim=0)
    #     loss=0
    #     for batch in range(self.hparams.batch_size):
    #         for frame in range(self.hparams.input_clip_length):
    #             for i,j in enumerate(y_true[batch][frame]):
    #                 if j:
    #                     loss-=np.log(y_pred[batch][i][frame].item())
    #     return loss
        
    def loss_function(self, output, y_true):
        act_loss = self.CEloss(output[0],torch.tensor( y_true[0],dtype=torch.long))
        subact_loss = self.CEloss(output[1],torch.tensor( y_true[1],dtype=torch.long))  
        aact_loss =self.BCEloss(torch.flatten(output[2]),torch.flatten( y_true[2]))
        loss =  aact_loss+subact_loss + act_loss
        return loss  # plot independently
    # 3: Data
    # was moved to the dataset python file

    # 4: Training Loop (this is where the magic happens)(training_step hook)
    def training_step(self, train_batch, batch_idx):
        x, y_true = train_batch  # x is the input image, y_true is the true label
        output = self.forward(x)
        
        # calculate loss
        train_loss = self.loss_function(output, y_true)

        # in DP mode (default) make sure if result is scalar, there's another dim in the beginning
        if self.trainer.use_dp or self.trainer.use_ddp2:
            train_loss = train_loss.unsqueeze(0)
            
        output=[output[0],output[1],output[2]]
        return {'loss': train_loss, 'y_true': y_true, 'output':output}
    # If using metrics in data parallel mode (dp), the metric update/logging should be done in the training_step_end
    # This is due to metric states else being destroyed after each forward pass, leading to wrong accumulation.
    
    def training_step_end(self, train_step_output):
        y0=np.array(self.softmax(train_step_output['output'][0]))
        y1=np.array(self.softmax(train_step_output['output'][1]))
        y2=np.array([self.sigmoid(j) for i in train_step_output['output'][2] for j in i ])
        activity_mAP=self.mAP(y0,np.array(train_step_output['y_true'][0]))
        sub_activity_mAP=self.mAP(y1,np.array(train_step_output['y_true'][1]))
        atomic_activity_mAP=self.mAP(y2,np.array(train_step_output['y_true'][2]))
        wandb.log({'Train_loss': train_step_output['loss'],#add specific losses correlated to act aact or subact
                   })
        return train_step_output

    # 5 Validation Loop
    def validation_step(self, val_batch, batch_idx):
        x, y_true = val_batch
        # y_true=self.reformat_true(y_true)
        output = self.forward(x)
        val_loss = self.loss_function(output , y_true)
        return {'loss': val_loss, 'y_true': y_true, 'output':output}

    # If using metrics in data parallel mode (dp), the metric update/logging should be done in the validation_step_end
    # This is due to metric states else being destroyed after each forward pass, leading to wrong accumulation.
    def validation_epoch_end(self, validation_step_outputs):
        all_preds = torch.stack(validation_step_outputs)
        y0=np.array(self.softmax(all_preds['output'][0]))
        y1=np.array(self.softmax(all_preds['output'][1]))
        y2=np.array([self.sigmoid(j) for i in all_preds['output'][2] for j in i ])
        self.log('Validation Loss', all_preds['loss'], on_epoch=True, on_step=False)
        
        wandb.log({'Validation_loss': all_preds['loss'],
                   #add specific losses correlated to act aact or subact

                })
        return all_preds

###___________________________________________________


    # 6 Test Loop
    def test_step(self, test_batch, batch_idx):
        x, y_true = test_batch
        y_pred = self(x)
        loss = self.loss_function(y_pred, y_true)
        return {'test_loss': loss, 'y_pred': y_pred, 'y_true': y_true}

    def test_step_end(self, test_step_output):
        Precision = self.Precision(test_step_output['y_pred'], test_step_output['y_true'])
        self.log('test_prec', Precision, on_step=True, on_epoch=True)
        return test_step_output
    
    

    @staticmethod
    def add_module_specific_args(parser):
        specific_args = get_argparser_group(title='Model options', parser=parser)
        return parser
