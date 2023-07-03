import pytorch_lightning as pl
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision
import pytorch_lightning as pl
import torch


class Faster_RCNN(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        num_classes = 141 #including background
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        
        # self.trace_model=torch.jit.trace(func=self.model,
        #                                  example_inputs=(torch.zeros(self.hparams.batch_size,
        #                                                              self.hparams.input_channels,
        #                                                              self.hparams.input_crop_size, 
        #                                                              self.hparams.input_crop_size),
        #                                                  {"boxes":[[0,0,312,312]],
        #                                                   "labels":[140]}),
        #                                  strict=False)
        # self.script_model=torch.jit.script(self.model)
    def forward(self, x):
        #reshaping required given the clip length and batch size, final format is (samples, channels, width, height)
        # input =torch.reshape(x,(-1,x.shape[-3],x.shape[-2],x.shape[-1]))
        output_model=self.model(x)
        # in_features = self.model.roi_heads.box_roi_pool.output
        boxes = output_model[0]['boxes']
        labels = output_model[0]['labels']
        scores= output_model[0]['scores']
        return boxes, labels, scores #,in_features
    
    def add_model_specific_args(parser):  # pragma: no cover
        model_specific_args = parser.add_argument_group(
            title='swin specific args options')
        model_specific_args.add_argument("--prediction_mode",
                                             default='last_label')
        return parser
# TO DO change header in order to have features class and bbox
