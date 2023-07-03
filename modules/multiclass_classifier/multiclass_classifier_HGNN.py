"""This is the module to the multiclass classification example.
It was run with the MNIST dataset and works with both MNIST models (mnist_ex1 and mnist_ex2 in the model folder)."""
import torch
from torch import optim
from pytorch_lightning.metrics import Precision
from pytorch_lightning.core.lightning import LightningModule
from utils.utils import get_argparser_group
import wandb
wandb.login()
import torch.nn as nn
from torchmetrics import AveragePrecision

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
        self.MLP
        run = wandb.init()
    # 1: Forward step (forward hook), Lightning calls this inside the training loop
    def forward(self, x):

        actor, frame, video = self.model(x) # the output is of the format:
            
        return actor, frame, video
        # TO DO: configure the loss and the rest
    
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
        return {'loss': train_loss, 'y_true': y_true, 'output':output,'y_true':y_true}
    #  F.log_softmax(torch.reshape(logits,(self.hparams.batch_size,18,self.hparams.input_clip_length)), dim=1)
    # If using metrics in data parallel mode (dp), the metric update/logging should be done in the training_step_end
    # This is due to metric states else being destroyed after each forward pass, leading to wrong accumulation.
    def training_step_end(self, train_step_output):
        y0=self.softmax(train_step_output['output'][0])
        y1=self.softmax(train_step_output['output'][1])
        y2=[self.sigmoid(j) for i in train_step_output['output'][2] for j in i ]
        print(  y0, train_step_output['y_true'][0])
        # AP_act= self.AP_act(y0,train_step_output['y_true'][0])
        # AP_subact= self.AP_subact(y1,train_step_output['y_true'][1])
        act_Precision = self.Precision(y0, train_step_output['y_true'][0])
        subact_Precision = self.Precision(y1, train_step_output['y_true'][1])
        aact_Precision = self.Precision(y2, torch.flatten(train_step_output['y_true'][2]))
        # train_Precision = np.mean((act_Precision.item(),subact_Precision.item(),aact_Precision.item()))
        # aact_AP= average_precision_score( F.one_hot(train_step_output['y_true'][0].cpu().detach().numpy(),self.hparams.activity_out_class)
        #                                  ,y0.cpu().detach().numpy())
        self.log('Train Loss', train_step_output['loss'], on_epoch=False, on_step=True)
        self.log('Train Aact Precision', aact_Precision, on_step=False, on_epoch=True)
        self.log('Train Subact Precision', subact_Precision, on_step=False, on_epoch=True)
        self.log('Train Activity Precision', act_Precision, on_step=False, on_epoch=True)
        # self.log('Train Precision',train_Precision)
        # self.log('activity AP',AP_act, on_epoch=False, on_step=True)
        # self.log('subactivity AP',AP_subact, on_epoch=False, on_step=True)

        wandb.log({'Train_loss': train_step_output['loss'],#add specific losses correlated to act aact or subact
                   'Train Aact Precision': aact_Precision,
                   'Train Subact Precision': subact_Precision,
                   'Train Activity Precision': act_Precision,
                #    'Train Precision':train_Precision,
                    # 'activity AP' : AP_act,
                    # 'subactivity AP': AP_subact
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
    def validation_step_end(self, val_step_output):
        y0=self.softmax(val_step_output['output'][0])
        y1=self.softmax(val_step_output['output'][1])
        y2=[self.sigmoid(j) for i in val_step_output['output'][2] for j in i ]
        # AP_act= self.AP_act(y0,val_step_output['y_true'][0])
        # AP_subact= self.AP_subact(y1,val_step_output['y_true'][1])
        act_Precision = self.Precision(y0, val_step_output['y_true'][0])
        subact_Precision = self.Precision(y1, val_step_output['y_true'][1])
        aact_Precision = self.Precision(y2, torch.flatten(val_step_output['y_true'][2]))
        # val_Precision = np.mean((act_Precision.item(),subact_Precision.item(),aact_Precision.item()))
        self.log('Validation Loss', val_step_output['loss'], on_epoch=False, on_step=True)
        self.log('Validation Aact Precision', aact_Precision, on_step=False, on_epoch=True)
        self.log('Validation Subact Precision', subact_Precision, on_step=False, on_epoch=True)
        self.log('Validation Activity Precision', act_Precision, on_step=False, on_epoch=True) # TO DO standardized logging Train/Loss_aact 
        # self.log('val_acc',val_Precision, on_epoch=True)
        # self.log('activity AP',AP_act, on_epoch=False, on_step=True)
        # self.log('subactivity AP',AP_subact, on_epoch=False, on_step=True)
        wandb.log({'Validation_loss': val_step_output['loss'],#add specific losses correlated to act aact or subact
                   'Validation Aact Precision': aact_Precision,
                   'Validation Subact Precision': subact_Precision,
                   'Validation Activity Precision': act_Precision,
                #    'val_acc':val_Precision,
                    # 'activity AP' : AP_act,
                    # 'subactivity AP': AP_subact
                })
        return val_step_output

    # 6 Test Loop
    def test_step(self, test_batch, batch_idx):
        x, y_true = test_batch
        y_pred = self(x)
        # print('y_pred shape is',y_pred.shape,'y_true shape is',y_true.shape)
        # print('y_pred looks like this',y_pred, 'and y_true looks like this',y_true)
        loss = self.loss_function(y_pred, y_true)
        return {'test_loss': loss, 'y_pred': y_pred, 'y_true': y_true}

    def test_step_end(self, test_step_output):
        Precision = self.Precision(test_step_output['y_pred'], test_step_output['y_true'])
        # self.log('test_loss', test_step_output['test_loss'], on_step=True, on_epoch=True)
        self.log('test_prec', Precision, on_step=True, on_epoch=True)
        return test_step_output
    
    

    @staticmethod
    def add_module_specific_args(parser):
        specific_args = get_argparser_group(title='Model options', parser=parser)
        return parser
