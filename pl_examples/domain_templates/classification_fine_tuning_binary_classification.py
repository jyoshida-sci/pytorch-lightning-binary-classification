"""Computer vision example on Transfer Learning.

This computer vision example illustrates how one could fine-tune a pre-trained
network (by default, a ResNet50 is used) using pytorch-lightning. For the sake
of this example, the 'cats and dogs dataset' (~60MB, see `DATA_URL` below) and
the proposed network (denoted by `TransferLearningModel`, see below) is
trained for 15 epochs. The training consists in three stages. From epoch 0 to
4, the feature extractor (the pre-trained network) is frozen except maybe for
the BatchNorm layers (depending on whether `train_bn = True`). The BatchNorm
layers (if `train_bn = True`) and the parameters of the classifier are trained
as a single parameters group with lr = 1e-2. From epoch 5 to 9, the last two
layer groups of the pre-trained network are unfrozen and added to the
optimizer as a new parameter group with lr = 1e-4 (while lr = 1e-3 for the
first parameter group in the optimizer). Eventually, from epoch 10, all the
remaining layer groups of the pre-trained network are unfrozen and added to
the optimizer as a third parameter group. From epoch 10, the parameters of the
pre-trained network are trained with lr = 1e-5 while those of the classifier
are trained with lr = 1e-4.

Note:
    See: https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
"""

import argparse
import os
from collections import OrderedDict, Counter
from pathlib import Path
from typing import Optional, Generator, Union

import torch
import torch.nn.functional as F
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torchvision import models
from torchvision import transforms
from torchvision.datasets import ImageFolder

import pytorch_lightning as pl
from pytorch_lightning import _logger as log
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
import datetime
import randaugment
from torchsampler import ImbalancedDatasetSampler

BN_TYPES = (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)


#  --- Utility functions ---

class MyEarlyStopping(EarlyStopping):

    def _run_early_stopping_check(self, trainer, pl_module):
        logs = trainer.callback_metrics
        if not self._validate_condition_metric(logs):
            return  # short circuit if metric not present

        current = logs.get(self.monitor)
        if not isinstance(current, torch.Tensor):
            current = torch.tensor(current, device=pl_module.device)

        if trainer.use_tpu and XLA_AVAILABLE:
            current = current.cpu()

        if self.monitor_op(current - self.min_delta, self.best_score):
            self.best_score = current
        else:
            should_stop =  current >= self.best_score * 1.15
            if bool(should_stop):
                self.stopped_epoch = trainer.current_epoch
                trainer.should_stop = True
        print(f"MyEarlyStopping, current:{current:.4f}, self.best:{self.best_score:.4f}, trainer.should_stop:{trainer.should_stop}\n")
        # stop every ddp process if any world process decides to stop
        self._stop_distributed_training(trainer, pl_module)


def _make_trainable(module: torch.nn.Module) -> None:
    """Unfreezes a given module.
    Args:
        module: The module to unfreeze
    """
    for param in module.parameters():
        param.requires_grad = True
    module.train()


def _recursive_freeze(module: torch.nn.Module,
                      train_bn: bool = True) -> None:
    """Freezes the layers of a given module.
    Args:
        module: The module to freeze
        train_bn: If True, leave the BatchNorm layers in training mode
    """
    children = list(module.children())
    if not children:
        if not (isinstance(module, BN_TYPES) and train_bn):
            for param in module.parameters():
                param.requires_grad = False
            module.eval()
        else:
            # Make the BN layers trainable
            _make_trainable(module)
    else:
        for child in children:
            _recursive_freeze(module=child, train_bn=train_bn)


def freeze(module: torch.nn.Module,
           n: Optional[int] = None,
           train_bn: bool = True) -> None:
    """Freezes the layers up to index n (if n is not None).
    Args:
        module: The module to freeze (at least partially)
        n: Max depth at which we stop freezing the layers. If None, all
            the layers of the given module will be frozen.
        train_bn: If True, leave the BatchNorm layers in training mode
    """
    children = list(module.children())
    n_max = len(children) if n is None else int(n)

    for child in children[:n_max]:
        _recursive_freeze(module=child, train_bn=train_bn)

    for child in children[n_max:]:
        _make_trainable(module=child)


def filter_params(module: torch.nn.Module,
                  train_bn: bool = True) -> Generator:
    """Yields the trainable parameters of a given module.
    Args:
        module: A given module
        train_bn: If True, leave the BatchNorm layers in training mode
    Returns:
        Generator
    """
    children = list(module.children())
    if not children:
        if not (isinstance(module, BN_TYPES) and train_bn):
            for param in module.parameters():
                if param.requires_grad:
                    yield param
    else:
        for child in children:
            for param in filter_params(module=child, train_bn=train_bn):
                yield param


def _unfreeze_and_add_param_group(module: torch.nn.Module,
                                  optimizer: Optimizer,
                                  lr: Optional[float] = None,
                                  train_bn: bool = True):
    """Unfreezes a module and adds its parameters to an optimizer."""
    _make_trainable(module)
    params_lr = optimizer.param_groups[0]['lr'] if lr is None else float(lr)
    optimizer.add_param_group(
        {'params': filter_params(module=module, train_bn=train_bn),
         'lr': params_lr / 10.,
         })


#  --- Pytorch-lightning module ---

class TransferLearningModel(pl.LightningModule):
    """Transfer Learning with pre-trained ResNet50.
    Args:
        hparams: Model hyperparameters
    """
    def __init__(self, hparams: argparse.Namespace) -> None:
        super().__init__()
        self.hparams = hparams
        self.build_model()


    def build_model(self):
        """Define model layers & loss."""
        model_func = getattr(models, self.hparams.backbone)
        backbone = model_func(pretrained=True)

        _layers = list(backbone.children())[:-1]
        self.feature_extractor = torch.nn.Sequential(*_layers)
        freeze(module=self.feature_extractor, train_bn=self.hparams.train_bn)
        _fc_layers = [torch.nn.Linear(2048, 256),
                      torch.nn.Linear(256, 32),
                      torch.nn.Linear(32, 1)]
        self.fc = torch.nn.Sequential(*_fc_layers)
        self.loss_func = F.binary_cross_entropy_with_logits


    def forward(self, x):
        """Forward pass. Returns logits."""
        x = self.feature_extractor(x)
        x = x.squeeze(-1).squeeze(-1)
        x = self.fc(x)
        return x


    def loss(self, labels, logits, my_pos_weight):
        return self.loss_func(input=logits, target=labels, pos_weight=my_pos_weight)


    def train(self, mode=True):
        super().train(mode=mode)
        epoch = self.current_epoch
        if epoch < self.hparams.milestone_a and mode:
            # feature extractor is frozen (except for BatchNorm layers)
            freeze(module=self.feature_extractor,
                   train_bn=self.hparams.train_bn)

        elif self.hparams.milestone_a <= epoch < self.hparams.milestone_b and mode:
            # Unfreeze last two layers of the feature extractor
            freeze(module=self.feature_extractor,
                   n=-2,
                   train_bn=self.hparams.train_bn)


    def on_epoch_start(self):
        """Use `on_epoch_start` to unfreeze layers progressively."""
        optimizer = self.trainer.optimizers[0]
        if self.current_epoch == self.hparams.milestone_a:
            _unfreeze_and_add_param_group(module=self.feature_extractor[-2:],
                                          optimizer=optimizer,
                                          train_bn=self.hparams.train_bn)
        elif self.current_epoch == self.hparams.milestone_b:
            _unfreeze_and_add_param_group(module=self.feature_extractor[:-2],
                                          optimizer=optimizer,
                                          train_bn=self.hparams.train_bn)

    def training_step(self, batch, batch_idx):
        x, y = batch
        #print(batch_idx, y)
        y_logits = self.forward(x)
        y_true = y.view((-1, 1)).type_as(x)
        y_bin = torch.ge(y_logits, 0)
        loss = self.loss(y_true, y_logits, self.weight_category['train'].type_as(x))
        num_correct = torch.eq(y_bin.view(-1), y_true.view(-1)).sum()
        tqdm_dict = {'train_loss': loss}
        output = OrderedDict({'loss': loss,
                              'num_correct': num_correct,
                              'log': tqdm_dict,
                              'progress_bar': tqdm_dict})
        return output

    def training_epoch_end(self, outputs):
        """Compute and log training loss and accuracy at the epoch level."""
        loss_mean = torch.stack([output['loss']
                                       for output in outputs]).mean()
        acc_mean = torch.stack([output['num_correct']
                                      for output in outputs]).sum().float()
        acc_mean /= (len(outputs) * self.hparams.batch_size)
        return {'log': {'0performance/loss_train_epoch': loss_mean,
                        '0performance/acc_train': acc_mean,
                        'step': self.current_epoch}}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_logits = self.forward(x)
        y_true = y.view((-1, 1)).type_as(x)
        y_bin = torch.ge(y_logits, 0)
        loss = self.loss(y_true, y_logits, self.weight_category['val'].type_as(x))
        num_correct = torch.eq(y_bin.view(-1), y_true.view(-1)).sum()
        return {'y':y,
                'y_logits':y_logits,
                'val_loss': loss,
                'num_correct': num_correct}

    def validation_epoch_end(self, outputs):
        """Compute and log validation loss and accuracy at the epoch level."""
        loss_mean = torch.stack([output['val_loss']
                                     for output in outputs]).mean()
        acc_mean = torch.stack([output['num_correct']
                                    for output in outputs]).sum().float()
        acc_mean /= (len(outputs) * self.hparams.batch_size)
        #
        list_y = torch.cat([output['y'] for output in outputs]).view(-1).tolist()
        list_y_logits = torch.cat([output['y_logits'] for output in outputs]).view(-1).tolist()
        aps = metrics.average_precision_score(list_y, list_y_logits)
        if self.current_epoch < 5:
            val_loss_ema = loss_mean
            self.val_loss_ema = loss_mean
        else:
            my_weight = 0.9
            val_loss_ema = self.val_loss_ema * my_weight + (1.0 - my_weight) * loss_mean
            self.val_loss_ema = val_loss_ema
        #
        return {'log': {'val_aps': torch.tensor(aps),
                        'val_loss': loss_mean,
                        'val_loss_ema': val_loss_ema,
                        '0performance/loss_val': loss_mean,
                        '0performance/acc_val': acc_mean,
                        'step': self.current_epoch}}

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_logits = self.forward(x)
        y_true = y.view((-1, 1)).type_as(x)
        y_bin = torch.ge(y_logits, 0)
        num_correct = torch.eq(y_bin.view(-1), y_true.view(-1)).sum()
        #
        specified_label= self.hparams.specified_label
        tensor_filled_specified_label = torch.full_like(y_bin, specified_label)
        num_specified_label_in_pred = torch.sum(y_bin == specified_label)
        num_specified_label_in_target = torch.sum(y_true == specified_label)
        true_or_false_in_pred = torch.eq(y_bin, tensor_filled_specified_label)
        true_or_false_in_target = torch.eq(y_true, tensor_filled_specified_label)
        num_specified_right = torch.mul(true_or_false_in_pred, true_or_false_in_target).sum()
        #
        output = OrderedDict({'y':y,
                              'y_logits':y_logits,
                              'num_correct': num_correct,
                              'num_specified_label_in_pred': num_specified_label_in_pred,
                              'num_specified_label_in_target': num_specified_label_in_target,
                              'num_specified_right': num_specified_right})
        return output


    def test_epoch_end(self, outputs):
        outputdir = self.trainer.weights_save_path
        print(outputdir)
        #
        list_y = torch.cat([output['y'] for output in outputs]).view(-1).tolist()
        list_y_logits = torch.cat([output['y_logits'] for output in outputs]).view(-1).tolist()
        #
        result_strs = []
        for i, l in enumerate(list_y):
            my_str = f"{list_y[i]} {list_y_logits[i]}\n"
            result_strs.append(my_str)
        with open(f'{outputdir}/../test_inference.txt', mode='w') as f:
            f.writelines(result_strs)
        #
        false_positive_rates, true_positive_rates, thresholds = metrics.roc_curve(list_y, list_y_logits)
        auroc = metrics.auc(false_positive_rates, true_positive_rates)
        #
        plt.rcParams["font.size"] = 15
        plt.plot(false_positive_rates, true_positive_rates, label = f'Area Under Curve = {auroc:0.2f}')
        plt.legend()
        plt.title('Receiver Operating Characteristic Curve')
        plt.xlabel('FPR: False positive rate')
        plt.ylabel('TPR: True positive rate')
        plt.grid()
        plt.savefig(f'{outputdir}/../sklearn_roc_curve.png')
        #
        precisions, recalls, thresholds = metrics.precision_recall_curve(list_y, list_y_logits)
        aps = metrics.average_precision_score(list_y, list_y_logits)
        plt.clf()
        plt.rcParams["font.size"] = 15
        plt.plot(recalls, precisions, label = f'Average precision score = {aps:0.2f}') # plt.plot(horizontal, vertical)
        plt.legend()
        plt.title('Precision-Recall Curve')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.grid()
        plt.savefig(f'{outputdir}/../sklearn_precision_recall_curve.png')
        #
        acc_mean = torch.stack([output['num_correct']
                                    for output in outputs]).sum().float()
        acc_mean /= (len(outputs) * self.hparams.batch_size)
        total_specified_label_in_pred = torch.stack([output['num_specified_label_in_pred']
                                    for output in outputs]).sum().float()
        total_specified_label_in_target = torch.stack([output['num_specified_label_in_target']
                                    for output in outputs]).sum().float()
        total_specified_right = torch.stack([output['num_specified_right']
                                    for output in outputs]).sum().float()
        recall = total_specified_right / total_specified_label_in_target
        precision = total_specified_right / total_specified_label_in_pred
        if (recall + precision)  < 0.0001:
            f_score = 0
        else:
            f_score = (2*recall*precision) / (recall + precision)
        
        return {'log': {'0performance/acc_test': acc_mean,
                        '0performance/auc': auroc,
                        '0performance/average_precision_score': aps,
                        '0performance/recall': recall,
                        '0performance/precision': precision,
                        '0performance/f_score': f_score,
                        'step': 0}}

    def configure_optimizers(self):
        optimizer = optim.Adam(filter(lambda p: p.requires_grad,
                                      self.parameters()),
                               lr=self.hparams.lr)
        scheduler = MultiStepLR(optimizer,
                                milestones=[self.hparams.milestone_a, self.hparams.milestone_b],
                                gamma=self.hparams.lr_scheduler_gamma)
        return [optimizer], [scheduler]


    def prepare_data(self):
        """Download images and prepare images datasets."""
        data_path = Path( self.hparams.root_data_path)
        resize = self.hparams.resize
        rand_augment_t = self.hparams.rand_augment_t
        rand_augment_n = self.hparams.rand_augment_n
        rand_augment_m = self.hparams.rand_augment_m
        # 2. Load the data + preprocessing & data augmentation
        train_dataset = ImageFolder(root=data_path.joinpath('train'),
                                    transform=transforms.Compose([
                                        transforms.Resize((resize, resize)),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.RandomVerticalFlip(),
                                        randaugment.RandAugment(rand_augment_t, rand_augment_n, rand_augment_m),
                                        transforms.ToTensor()
                                    ]))
        val_dataset = ImageFolder(root=data_path.joinpath('val'),
                                    transform=transforms.Compose([
                                        transforms.Resize((resize, resize)),
                                        transforms.ToTensor()
                                    ]))
        test_dataset = ImageFolder(root=data_path.joinpath('test'),
                                    transform=transforms.Compose([
                                        transforms.Resize((resize, resize)),
                                        transforms.ToTensor()
                                    ]))
        self.dataset = {'train': train_dataset,
                        'val': val_dataset,
                        'test': test_dataset}
        self.weight_category = {}
        for k, d in self.dataset.items():
            n_items = list(Counter(d.targets).values())
            weights = [len(train_dataset.targets)/x for x in n_items]
            normalized = [x/max(weights) for x in weights]
            ratio = normalized[1]/normalized[0]
            self.weight_category[k] = torch.FloatTensor([ratio])


    def __dataloader(self, mode):
        loader = DataLoader(dataset=self.dataset[mode],
                            batch_size=self.hparams.batch_size,
                            num_workers=self.hparams.num_workers,
                            #shuffle=True if mode=='train' else False)
                            sampler=ImbalancedDatasetSampler(self.dataset[mode]) if mode=='train' else None)
        return loader

    def train_dataloader(self):
        log.info('Training data loaded.')
        return self.__dataloader(mode='train')

    def val_dataloader(self):
        log.info('Validation data loaded.')
        return self.__dataloader(mode='val')

    def test_dataloader(self):
        log.info('Validation data loaded.')
        return self.__dataloader(mode='test')


    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser])
        parser.add_argument('--backbone',
                            default='resnet50',
                            type=str,
                            metavar='BK',
                            help='Name (as in ``torchvision.models``) of the feature extractor')
        parser.add_argument('--max_epochs',
                            default=1000,
                            type=int,
                            metavar='MAXEPOCH',
                            help='max number of epochs')
        parser.add_argument('--min_epochs',
                            default=20,
                            type=int,
                            metavar='MINEPOCH',
                            help='min number of epochs')
        parser.add_argument('--batch_size',
                            default=96,
                            type=int,
                            metavar='B',
                            help='batch size')
        parser.add_argument('--gpu',
                            type=str,
                            default='0',
                            help='gpu ID to use')
        parser.add_argument('--lr',
                            '--learning_rate',
                            default=3e-4,
                            type=float,
                            metavar='LR',
                            help='initial learning rate',
                            dest='lr')
        parser.add_argument('--lr_scheduler_gamma',
                            default=1e-1,
                            type=float,
                            metavar='LRG',
                            help='Factor by which the learning rate is reduced at each milestone')
        parser.add_argument('--num_workers',
                            default=4,
                            type=int,
                            metavar='W',
                            help='number of CPU workers')
        parser.add_argument('--train_bn',
                            default=True,
                            type=bool,
                            metavar='TB',
                            help='Whether the BatchNorm layers should be trainable')
        parser.add_argument('--milestone_a',
                            default=50,
                            type=int,
                            metavar='MA',
                            help='milestone_a')
        parser.add_argument('--milestone_b',
                            default= 100,
                            type=int,
                            metavar='MB',
                            help='milestone_b')
        parser.add_argument('--resize',
                            default=224,
                            type=int,
                            metavar='RES',
                            help='resize')
        parser.add_argument('--rand_augment_t',
                            default=4,
                            type=int,
                            metavar='RAND_AUG_T',
                            help='rand_augment_t')
        parser.add_argument('--rand_augment_n',
                            default=2,
                            type=int,
                            metavar='RAND_AUG_N',
                            help='rand_augment_n')
        parser.add_argument('--rand_augment_m',
                            default=9,
                            type=int,
                            metavar='RAND_AUG_M',
                            help='rand_augment_m')
        parser.add_argument('--specified_label',
                            default=1,
                            type=int,
                            metavar='SPEL',
                            help='specified label for recall, precison, and f-score')
        return parser


def main(hparams: argparse.Namespace) -> None:
    """Train the model.
    Args:
        hparams: Model hyper-parameters
    """
    model = TransferLearningModel(hparams)
    my_checkpoint_callback = ModelCheckpoint(
        monitor='val_loss_ema',
        mode='min',
        )
    my_early_stop_callback = MyEarlyStopping(
        monitor='val_loss_ema',
        min_delta=0.0,
        verbose=True,
        mode='min'
        )
    trainer = pl.Trainer(
        default_root_dir=f"classification_fine_tuning/{hparams.out_dir_name}", # output
        weights_summary=None,
        show_progress_bar=True,
        num_sanity_val_steps=0,
        gpus=hparams.gpu,
        checkpoint_callback=my_checkpoint_callback,
        early_stop_callback=my_early_stop_callback,
        min_epochs= hparams.min_epochs,
        max_epochs=hparams.max_epochs)
    trainer.fit(model)

    #my_map_location = f'cuda:{hparams.gpu}'
    #print(my_map_location)
    #model = model.load_from_checkpoint(
    #                        trainer.checkpoint_callback.kth_best_model,
    #                        map_location= my_map_location)
    #trainer.test(model)


def get_args() -> argparse.Namespace:
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument('--root_data_path',
                               metavar='DIR',
                               type=str,
                               #default='dataset/cats_and_dogs_filtered_orig',
                               default='my_dataset',
                               help='Root directory of dataset. train, val, and test. category must be 0_negative and 1_positive')
    parent_parser.add_argument('--out_dir_name',
                               metavar='DIR',
                               type=str,
                               default='output_my_dataset',
                               help='Output directory: classification_fine_tuning/???/lightning_logs/version_?')
    parser = TransferLearningModel.add_model_specific_args(parent_parser)
    return parser.parse_args()


if __name__ == '__main__':

    main(get_args())
