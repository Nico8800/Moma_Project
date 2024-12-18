"""Taken from: TeCNO gitlab repo master branch"""
import configargparse
from pathlib import Path
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from utils.plx_logger import PolyaxonLogger
from utils.utils import (
    argparse_summary,
    get_class_by_path,
)
from utils.configargparse_arguments import build_configargparser
from datetime import datetime
import warnings
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
warnings.filterwarnings("ignore", category=UserWarning)
from pytorch_lightning.loggers import WandbLogger

def train(hparams, ModuleClass, ModelClass, DatasetClass, logger):
    """
    Main training routine specific for this project
    :param hparams:
    """
    # ------------------------
    # 1 INIT LIGHTNING MODEL
    # ------------------------
    # load model
    print('In train.py: Loading model...')
    model = ModelClass(hparams=hparams)
    print('...done.')
    # load dataset
    print('In train.py: Loading dataset...')
    dataset = DatasetClass(hparams=hparams)
    print('...done.')
    # load module
    print('In train.py: Loading module...')
    module = ModuleClass(hparams, model)
    print('...done.')

    # ------------------------
    # 3 INIT TRAINER --> continues training
    # ------------------------
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{hparams.output_path}/checkpoints/",
        save_top_k=hparams.save_top_k,
        verbose=True,
        monitor=hparams.early_stopping_metric,
        mode='max',
        filename=f'{{epoch}}-{{{hparams.early_stopping_metric}:.2f}}'
    )
    early_stop_callback = EarlyStopping(
        monitor=hparams.early_stopping_metric,
        min_delta=0.00,
        patience=3,
        mode='max')

    trainer = Trainer(accelerator='cuda',
                        gpus=hparams.gpus,
                        precision=16,
                        logger=logger,
                        # fast_dev_run: if true, runs one training and one validation batch
                        fast_dev_run=hparams.fast_dev_run,
                        # min_epochs: forces training to a minimum number of epochs
                        min_epochs=hparams.min_epochs,
                        # max_epochs: limits training to a maximum number of epochs
                        max_epochs=hparams.max_epochs,
                        # saves the state of the last training epoch (all model parameters)
                        # checkpoint_callback=True,
                        resume_from_checkpoint=hparams.resume_from_checkpoint,
                        callbacks=[early_stop_callback, checkpoint_callback],
                        # weights_summary='full',
                        # runs a certain number of validation steps before training to catch bugs
                        # num_sanity_val_steps=hparams.num_sanity_val_steps,
                        log_every_n_steps=hparams.log_every_n_steps,
                        # auto_lr_find: if true, will find a learning rate that optimizes initial learning for faster convergence
                        auto_lr_find=True,
                        # auto_scale_batch_size: if true, will initially find the largest batch size that fits into memory
                        auto_scale_batch_size=True,
                        limit_train_batches=hparams.limit_val_batches,  # use 0.15 - 0.2 for Polyaxon, use 0.03 to avoid memory error on Anna's computer
                        limit_val_batches=hparams.limit_val_batches,
                        limit_test_batches=hparams.limit_test_batches,# use 0.35 - 0.4for Polyaxon, use 0.05 to avoid memory error on
                        default_root_dir="/home/nicolasg/IDP/checkpoints"                    # Nicolas's computer
                )
    # ------------------------
    # 4 START TRAINING
    # ------------------------
    print('Starting training...')
    trainer.fit(module, dataset)
    # print(hparams.run_test)
    if hparams.run_test:
        print(
            f"Best: {checkpoint_callback.best_model_score} | monitor: {checkpoint_callback.monitor} "
            f"| path: {checkpoint_callback.best_model_path}"
            f"\nTesting..."
        )
        trainer.test(ckpt_path=checkpoint_callback.best_model_path, verbose=True)


if __name__ == "__main__":
    # ------------------------
    # TRAINING ARGUMENTS
    # ------------------------
    # these are project-wide arguments

    root_dir = Path(__file__).parent
    parser = configargparse.ArgParser(
        config_file_parser_class=configargparse.YAMLConfigFileParser)
    parser.add_argument('-c', is_config_file=True, help='config file path')

    parser, hparams = build_configargparser(parser)
    # each LightningModule defines arguments relevant to it
    # ------------------------
    # LOAD MODULE
    # ------------------------
    module_path = f"modules.{hparams.module}"
    ModuleClass = get_class_by_path(module_path)
    parser = ModuleClass.add_module_specific_args(parser)
    # ------------------------
    # LOAD MODEL
    # ------------------------
    model_path = f"models.{hparams.model}"
    ModelClass = get_class_by_path(model_path)
    parser = ModelClass.add_model_specific_args(parser)
    # ------------------------
    # LOAD DATASET
    # ------------------------
    dataset_path = f"datasets.{hparams.dataset}"
    DatasetClass = get_class_by_path(dataset_path)
    parser = DatasetClass.add_dataset_specific_args(parser)
    # ------------------------
    #  PRINT PARAMS & INIT LOGGER
    # ------------------------
    hparams = parser.parse_args()
    # setup logging
    exp_name = (
            hparams.module.split(".")[-1]
            + "_"
            + hparams.dataset.split(".")[-1]
            + "_"
            + hparams.model.replace(".", "_")
    )
    if hparams.on_polyaxon:
        plx_logger = PolyaxonLogger(hparams)
        hparams.output_path = plx_logger.output_path
        hparams = plx_logger.hparams
        hparams.name = plx_logger.experiment.experiment_id + "_" + exp_name
    else:
        date_str = datetime.now().strftime("%y%m%d-%H%M%S_")
        hparams.name = date_str + exp_name
        hparams.output_path = Path(hparams.output_path).absolute() / hparams.name

    tb_logger = TensorBoardLogger(hparams.output_path, name='tb', log_graph=True)
    wandb_logger = WandbLogger(project="MOMA")
    
    argparse_summary(hparams, parser)
    print('Output path: ', hparams.output_path)

    loggers = [tb_logger, wandb_logger, plx_logger] if hparams.on_polyaxon else [tb_logger,wandb_logger]

    # ---------------------
    # RUN TRAINING
    # ---------------------
    train(hparams, ModuleClass, ModelClass, DatasetClass, loggers)
