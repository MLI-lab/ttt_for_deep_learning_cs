"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import pathlib
import random

import numpy as np
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.logging import TestTubeLogger
from torch.nn import functional as F
from torch.optim import RMSprop

from .common.args import Args
from .common.subsample import create_mask_for_mask_type
from .data import transforms
from .mri_model import MRIModel
from .unet_model import UnetModel

#torch.backends.cudnn.enabled = True
#torch.backends.cudnn.benchmark = True
#torch.cuda.set_device(3)

import os
#os.environ['CUDA_VISIBLE_DEVICES']='3'

#_Trainer__set_random_port()

class DataTransform:
    """
    Data Transformer for training U-Net models.
    """

    def __init__(self, resolution, which_challenge, mask_func=None, use_seed=True):
        """
        Args:
            mask_func (common.subsample.MaskFunc): A function that can create a mask of
                appropriate shape.
            resolution (int): Resolution of the image.
            which_challenge (str): Either "singlecoil" or "multicoil" denoting the dataset.
            use_seed (bool): If true, this class computes a pseudo random number generator seed
                from the filename. This ensures that the same mask is used for all the slices of
                a given volume every time.
        """
        if which_challenge not in ('singlecoil', 'multicoil'):
            raise ValueError(f'Challenge should either be "singlecoil" or "multicoil"')
        self.mask_func = mask_func
        self.resolution = resolution
        self.which_challenge = which_challenge
        self.use_seed = use_seed

    def __call__(self, kspace, target, attrs, fname, slice):
        """
        Args:
            kspace (numpy.array): Input k-space of shape (num_coils, rows, cols, 2) for multi-coil
                data or (rows, cols, 2) for single coil data.
            target (numpy.array): Target image
            attrs (dict): Acquisition related information stored in the HDF5 object.
            fname (str): File name
            slice (int): Serial number of the slice.
        Returns:
            (tuple): tuple containing:
                image (torch.Tensor): Zero-filled input image.
                target (torch.Tensor): Target image converted to a torch Tensor.
                mean (float): Mean value used for normalization.
                std (float): Standard deviation value used for normalization.
        """
        kspace = transforms.to_tensor(kspace)
        # Apply mask
        if self.mask_func:
            seed = None if not self.use_seed else tuple(map(ord, fname))
            masked_kspace, mask = transforms.apply_mask(kspace, self.mask_func, seed)
        else:
            masked_kspace = kspace

        # Inverse Fourier Transform to get zero filled solution
        image = transforms.ifft2(masked_kspace)
        # Crop input image to given resolution if larger
        smallest_width = min(self.resolution, image.shape[-2])
        smallest_height = min(self.resolution, image.shape[-3])
        if target is not None:
            smallest_width = min(smallest_width, target.shape[-1])
            smallest_height = min(smallest_height, target.shape[-2])
        crop_size = (smallest_height, smallest_width)
        
        ######################################## NO CROP ################################################## MZD
        '''
        image = transforms.complex_center_crop(image, crop_size) 
        
        ##############
        temp = image.clone()
        temp = torch.zeros([image.shape[0],self.resolution,self.resolution,image.shape[-1]])
        width_diff = (self.resolution-image.shape[-2])//2
        height_diff = (self.resolution-image.shape[-3])//2

        ws = width_diff + int(image.shape[-2]%2)
        we = temp.shape[-2]-width_diff
        #print(ws,we,width_diff,image.shape)
        hs = height_diff + int(image.shape[-3]%2)
        he = temp.shape[-3]-height_diff

        temp[:,hs:he,ws:we,:] = image
        # Absolute value
        image = transforms.complex_abs(temp) ############
        '''
        ################################################################################################### MZD
        
        # Apply Root-Sum-of-Squares if multicoil data
        if self.which_challenge == 'multicoil':
            image = transforms.root_sum_of_squares(image)
        image = torch.moveaxis(image , 2 , 0) ############################# MZD
        # Normalize input
        image, mean, std = transforms.normalize_instance(image, eps=1e-11)
        image = image.clamp(-6, 6)
        #print(image.shape)
        # Normalize target
        if target is not None:
            target = transforms.ifft2(kspace) ############################# MZD
            target = torch.moveaxis( transforms.root_sum_of_squares(target) , 2 , 0) ############################# MZD
            #print(target.shape)
            #im = transform.complex_abs(kspace)
            ############################### NO CROP - TARGET IS IFFT2(KSPACE) ##################################### MZD
            '''
            target = transforms.to_tensor(target)
            target = transforms.center_crop(target, crop_size) 
            #print(target.shape)
            ##############
            temp = target.clone()
            temp = torch.zeros([self.resolution,self.resolution])
            width_diff = (self.resolution-target.shape[-1])//2
            height_diff = (self.resolution-target.shape[-2])//2

            ws = width_diff + int(target.shape[-1]%2)
            we = temp.shape[-1]-width_diff
            hs = height_diff + int(target.shape[-2]%2)
            he = temp.shape[-2]-height_diff

            temp[hs:he,ws:we] = target
            ###############
            '''
            ##################################################################################### MZD
            target = transforms.normalize(target, mean, std, eps=1e-11)
            target = target.clamp(-6, 6)
        else:
            target = torch.Tensor([0])
        return image, target, mean, std, fname, slice


class UnetMRIModel(MRIModel):
    def __init__(self, hparams):
        super().__init__(hparams)
        self.unet = UnetModel(
            in_chans=hparams.in_chans, ############################################################## MZD
            out_chans=hparams.in_chans, ############################################################## MZD
            chans=hparams.num_chans,
            num_pool_layers=hparams.num_pools,
            drop_prob=hparams.drop_prob
        )

    def forward(self, input):
        return self.unet(input) #(input.unsqueeze(1)).squeeze(1) ############## MZD

    def training_step(self, batch, batch_idx):
        input, target, mean, std, _, _ = batch
        #print(input.shape,target.shape)
        output = self.forward(input)
        loss = F.l1_loss(output, target)
        logs = {'loss': loss.item()}
        return dict(loss=loss, log=logs)

    def validation_step(self, batch, batch_idx):
        input, target, mean, std, fname, slice = batch
        output = self.forward(input)
        #print(output.shape)
        mean = mean.unsqueeze(1).unsqueeze(2)
        std = std.unsqueeze(1).unsqueeze(2)
        return {
            'fname': fname,
            'slice': slice,
            'output': (output * std + mean).cpu().numpy(),
            'target': (target * std + mean).cpu().numpy(),
            'val_loss': F.l1_loss(output, target),
        }

    def test_step(self, batch, batch_idx):
        input, _, mean, std, fname, slice = batch
        output = self.forward(input)
        mean = mean.unsqueeze(1).unsqueeze(2)
        std = std.unsqueeze(1).unsqueeze(2)
        return {
            'fname': fname,
            'slice': slice,
            'output': (output * std + mean).cpu().numpy(),
        }

    def configure_optimizers(self):
        optim = RMSprop(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optim, self.hparams.lr_step_size, self.hparams.lr_gamma)
        return [optim], [scheduler]

    def train_data_transform(self):
        mask = create_mask_for_mask_type(self.hparams.mask_type, self.hparams.center_fractions,
                                         self.hparams.accelerations)
        return DataTransform(self.hparams.resolution, self.hparams.challenge, mask, use_seed=False)

    def val_data_transform(self):
        mask = create_mask_for_mask_type(self.hparams.mask_type, self.hparams.center_fractions,
                                         self.hparams.accelerations)
        return DataTransform(self.hparams.resolution, self.hparams.challenge, mask)

    def test_data_transform(self):
        return DataTransform(self.hparams.resolution, self.hparams.challenge)

    @staticmethod
    def add_model_specific_args(parser):
        parser.add_argument('--num-pools', type=int, default=4, help='Number of U-Net pooling layers')
        parser.add_argument('--drop-prob', type=float, default=0.0, help='Dropout probability')
        parser.add_argument('--num-chans', type=int, default=32, help='Number of U-Net channels')
        parser.add_argument('--batch-size', default=16, type=int, help='Mini batch size')
        parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
        parser.add_argument('--lr-step-size', type=int, default=40,
                            help='Period of learning rate decay')
        parser.add_argument('--lr-gamma', type=float, default=0.1,
                            help='Multiplicative factor of learning rate decay')
        parser.add_argument('--weight-decay', type=float, default=0.,
                            help='Strength of weight decay regularization')
        parser.add_argument('--mask_type',default='random')
        parser.add_argument('--in-chans', type=int, default=2, help='Number of U-Net input (and output) channels')
        return parser


def create_trainer(args, logger):
    return Trainer(
        #num_nodes=1,
        logger=logger,
        default_save_path=args.exp_dir,
        checkpoint_callback=True,
        max_nb_epochs=args.num_epochs,
        gpus=args.gpus,
        distributed_backend='ddp',
        check_val_every_n_epoch=1,
        val_check_interval=1.,
        early_stop_callback=False
    )


def main(args):
    if args.mode == 'train':
        load_version = 0 if args.resume else None
        logger = TestTubeLogger(save_dir=args.exp_dir, name=args.exp, version=load_version)
        trainer = create_trainer(args, logger)
        model = UnetMRIModel(args)
        trainer.fit(model)
    else:  # args.mode == 'test'
        assert args.checkpoint is not None
        model = UnetMRIModel.load_from_checkpoint(str(args.checkpoint))
        model.hparams.sample_rate = 1.
        trainer = create_trainer(args, logger=False)
        trainer.test(model)


if __name__ == '__main__':
    parser = Args()
    parser.add_argument('--mode', choices=['train', 'test'], default='train')
    parser.add_argument('--num-epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--exp-dir', type=pathlib.Path, default='experiments',
                        help='Path where model and results should be saved')
    parser.add_argument('--exp', type=str, help='Name of the experiment')
    parser.add_argument('--checkpoint', type=pathlib.Path,
                        help='Path to pre-trained model. Use with --mode test')
    parser.add_argument('--resume', action='store_true',
                        help='If set, resume the training from a previous model checkpoint. ')
    parser = UnetMRIModel.add_model_specific_args(parser)
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    main(args)
