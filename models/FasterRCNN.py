import pytorch_lightning as pl
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, fasterrcnn_resnet50_fpn
import pytorch_lightning as pl
import torch

class FasterRCNN(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.pretrained=True
        self.pretrained_backbone= True
        self.trainable_backbone_layers=3
        self.model = fasterrcnn_resnet50_fpn(
                pretrained=self.pretrained,
                pretrained_backbone=self.pretrained_backbone,
                trainable_backbone_layers=self.trainable_backbone_layers,
            )
        num_classes = 141 #including background
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    def forward(self, x):
        self.model.eval()
        return self.model(torch.reshape(x,(x.shape[-4],x.shape[-1],x.shape[-3],x.shape[-2])) )   

    
    def add_model_specific_args(parser):  # pragma: no cover
        model_specific_args = parser.add_argument_group(
            title='swin specific args options')
        model_specific_args.add_argument("--prediction_mode",
                                             default='last_label')
        return parser
# TO DO change header in order to have features class and bbox
