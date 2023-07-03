"""This is the module to the multiclass classification example.
It was run with the MNIST dataset and works with both MNIST models (mnist_ex1 and mnist_ex2 in the model folder)."""
import torch
from torch import optim
import torch.nn.functional as F
from pytorch_lightning.metrics import Accuracy
from pytorch_lightning.core.lightning import LightningModule
from utils.utils import get_argparser_group
import wandb
wandb.login()
import numpy as np

class MultiClassClassification(LightningModule):
    def __init__(self, hparams, model):
        super(MultiClassClassification, self).__init__()
        self.hparams = hparams
        self.model = model
        self.example_input_array = torch.zeros(self.hparams.batch_size, self.hparams.input_channels, 28, 28)
        self.accuracy = Accuracy()

    # 1: Forward step (forward hook), Lightning calls this inside the training loop
    def forward(self, x):
        x = self.model.forward(x)
        return x

    # 2: Optimizer (configure_optimizers hook)
    # see https://pytorch-lightning.readthedocs.io/en/latest/optimizers.html
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        # where T_max is the maximum number of iterations
        return [optimizer], [scheduler]

    def loss_function(self, y_pred, y_true):
        return F.cross_entropy(y_pred, y_true)

    # 3: Data
    # was moved to the dataset python file

    # 4: Training Loop (this is where the magic happens)(training_step hook)
    def training_step(self, train_batch, batch_idx):
        x, y_true = train_batch  # x is the input image, y_true is the true label
        # display a few images
        if self.current_epoch == 0 and batch_idx == 0:
            self.logger.experiment[0].add_images('Epoch {} Batch {}'.format(self.current_epoch, batch_idx), x, 0)
        # forward pass
        y_pred = self.forward(x)
        # calculate loss
        train_loss = self.loss_function(y_pred, y_true)
        # in DP mode (default) make sure if result is scalar, there's another dim in the beginning
        if self.trainer.use_dp or self.trainer.use_ddp2:
            train_loss = train_loss.unsqueeze(0)
        return {'loss': train_loss, 'y_true': y_true, 'y_pred': y_pred}

    # If using metrics in data parallel mode (dp), the metric update/logging should be done in the training_step_end
    # This is due to metric states else being destroyed after each forward pass, leading to wrong accumulation.
    def training_step_end(self, train_step_output):
        accuracy = self.accuracy(train_step_output['y_pred'], train_step_output['y_true'])
        self.log('Train Loss', train_step_output['loss'], on_epoch=False, on_step=True)
        self.log('Train Accuracy', accuracy, on_step=False, on_epoch=True)
        wandb.log({'train_loss': train_step_output['loss'],'Train Accuracy': accuracy})
        return train_step_output

    # 5 Validation Loop
    def validation_step(self, val_batch, batch_idx):
        x, y_true = val_batch
        y_pred = self.forward(x)
        val_loss = self.loss_function(y_pred, y_true)
        return {'val_loss': val_loss, 'y_true': y_true, 'y_pred': y_pred}

    # If using metrics in data parallel mode (dp), the metric update/logging should be done in the validation_step_end
    # This is due to metric states else being destroyed after each forward pass, leading to wrong accumulation.
    def validation_step_end(self, val_step_output):
        accuracy = self.accuracy(val_step_output['y_pred'], val_step_output['y_true'])
        self.log('val_loss', val_step_output['val_loss'], on_step=True, on_epoch=True, prog_bar=True)
        self.log('val_acc', accuracy, on_step=False, on_epoch=True, prog_bar=True)
        wandb.log({'val_loss': val_step_output['val_loss'],'Val Accuracy': accuracy})
        return val_step_output

    # 6 Test Loop
    def test_step(self, test_batch, batch_idx):
        x, y_true = test_batch
        y_pred = self(x)
        print('y_pred shape is',y_pred.shape,'y_true shape is',y_true.shape)
        print('y_pred looks like this',y_pred, 'and y_true looks like this',y_true)
        loss = self.loss_function(y_pred, y_true)
        return {'test_loss': loss, 'y_pred': y_pred, 'y_true': y_true}

    def test_step_end(self, test_step_output):
        accuracy = self.accuracy(test_step_output['y_pred'], test_step_output['y_true'])
        # self.log('test_loss', test_step_output['test_loss'], on_step=True, on_epoch=True)
        self.log('test_acc', accuracy, on_step=True, on_epoch=True)
        return test_step_output

    @staticmethod
    def add_module_specific_args(parser):
        specific_args = get_argparser_group(title='Model options', parser=parser)
        specific_args.add_argument('--input_channels', default=3, type=int,
                                   help='number of input channels (default: 3)')
        return parser
