from argparse import ArgumentParser
from typing import Any, Optional, Union

import torch
from pytorch_lightning import LightningModule, Trainer, seed_everything

from pl_bolts.utils import _TORCHVISION_AVAILABLE
# from pl_bolts.utils.stability import under_review
from pl_bolts.utils.warnings import warn_missing_pkg

if _TORCHVISION_AVAILABLE:
    from torchvision.models.detection.faster_rcnn import FasterRCNN as torchvision_FasterRCNN
else:  # pragma: no cover
    warn_missing_pkg("torchvision")
from utils.metrics import _evaluate_iou
from utils.utils import get_argparser_group


class FasterRCNN(LightningModule):
    """PyTorch Lightning implementation of `Faster R-CNN: Towards Real-Time Object Detection with Region Proposal
    Networks <https://arxiv.org/abs/1506.01497>`_.
    Paper authors: Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun
    Model implemented by:
        - `Teddy Koker <https://github.com/teddykoker>`
    During training, the model expects both the input tensors, as well as targets (list of dictionary), containing:
        - boxes (`FloatTensor[N, 4]`): the ground truth boxes in `[x1, y1, x2, y2]` format.
        - labels (`Int64Tensor[N]`): the class label for each ground truh box
    CLI command::
        # PascalVOC
        python faster_rcnn_module.py --gpus 1 --pretrained True
    """

    def __init__(
        self,
        hparams,
        model):
        if not _TORCHVISION_AVAILABLE:  # pragma: no cover
            raise ModuleNotFoundError("You want to use `torchvision` which is not installed yet.")

        super().__init__()
        self.model=model
        self.learning_rate = hparams.learning_rate
        self.num_classes = 141


        

    def training_step(self, batch, batch_idx):

        images, targets = batch
        targets = [{k: v for k, v in t.items()} for t in targets]

        # fasterrcnn takes both images and targets for training, returns
        loss_dict = self.model(images, targets)
        loss = sum(loss for loss in loss_dict.values())
        return {"loss": loss, "log": loss_dict}

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        # fasterrcnn takes only images for eval() mode
        outs = self.model(images)
        iou = torch.stack([_evaluate_iou(t, o) for t, o in zip(targets, outs)]).mean()
        return {"val_iou": iou}

    def validation_epoch_end(self, outs):
        avg_iou = torch.stack([o["val_iou"] for o in outs]).mean()
        logs = {"val_iou": avg_iou}
        return {"avg_val_iou": avg_iou, "log": logs}

    def configure_optimizers(self):
        return torch.optim.SGD(
            self.model.parameters(),
            lr=self.learning_rate,
            momentum=0.9,
            weight_decay=0.005,
        )

    @staticmethod
    def add_module_specific_args(parser):
        specific_args = get_argparser_group(title='Model options', parser=parser)
        return parser
