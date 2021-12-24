"""
Description:
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2021-05-10 19:31:03
LastEditors: Jiaqi Gu (jqgu@utexas.edu)
LastEditTime: 2021-11-29 22:39:02
"""

from typing import Tuple

import torch
import torch.nn as nn
import torchvision
from pyutils.config import configs
from pyutils.datasets import get_dataset
from pyutils.typing import DataLoader, Optimizer, Scheduler
from torch.types import Device
from torchonn.devices import *
from torchvision import datasets, transforms

from core.models import *

__all__ = ["make_dataloader", "make_model", "make_optimizer", "make_scheduler", "make_criterion"]


def make_dataloader() -> Tuple[DataLoader, DataLoader]:
    transform = configs.dataset.transform
    img_height, img_width = configs.dataset.img_height, configs.dataset.img_width
    dataset_dir = configs.dataset.root
    if configs.dataset.name == "cifar10":
        if transform == "basic":
            t = []
            if (img_height, img_width) != (32, 32):
                t.append(transforms.Resize((img_height, img_width), interpolation=2))
            transform_test = transform_train = transforms.Compose(t + [transforms.ToTensor()])

        else:
            transform_train = transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.Resize((img_height, img_width), interpolation=2),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ]
            )

            transform_test = transforms.Compose(
                [
                    transforms.Resize((img_height, img_width), interpolation=2),
                    transforms.ToTensor(),
                    # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ]
            )
        train_dataset = datasets.CIFAR10(dataset_dir, train=True, download=True, transform=transform_train)

        validation_dataset = datasets.CIFAR10(dataset_dir, train=False, transform=transform_test)
    elif configs.dataset.name == "cifar100":
        if transform == "basic":
            t = []
            if (img_height, img_width) != (32, 32):
                t.append(transforms.Resize((img_height, img_width), interpolation=2))
            transform_test = transform_train = transforms.Compose(t + [transforms.ToTensor()])
        else:
            # CIFAR100_TRAIN_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
            # CIFAR100_TRAIN_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
            transform_train = transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.Resize((img_height, img_width), interpolation=2),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    # transforms.Normalize(CIFAR100_TRAIN_MEAN, CIFAR100_TRAIN_STD),
                ]
            )

            transform_test = transforms.Compose(
                [
                    transforms.Resize((img_height, img_width), interpolation=2),
                    transforms.ToTensor(),
                    # transforms.Normalize(CIFAR100_TRAIN_MEAN, CIFAR100_TRAIN_STD),
                ]
            )
        train_dataset = datasets.CIFAR100(dataset_dir, train=True, download=True, transform=transform_train)

        validation_dataset = datasets.CIFAR100(dataset_dir, train=False, transform=transform_test)
    elif configs.dataset.name == "svhn":
        if transform == "basic":
            t = []
            if (img_height, img_width) != (32, 32):
                t.append(transforms.Resize((img_height, img_width), interpolation=2))
            transform_test = transform_train = transforms.Compose(t + [transforms.ToTensor()])

        else:
            # SVHN_TRAIN_MEAN = (0.4377, 0.4438, 0.4728)
            # SVHN_TRAIN_STD = (0.1980, 0.2010, 0.1970)
            transform_train = transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.Resize((img_height, img_width), interpolation=2),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    # transforms.Normalize(SVHN_TRAIN_MEAN, SVHN_TRAIN_STD),
                ]
            )

            transform_test = transforms.Compose(
                [
                    transforms.Resize((img_height, img_width), interpolation=2),
                    transforms.ToTensor(),
                    # transforms.Normalize(SVHN_TRAIN_MEAN, SVHN_TRAIN_STD),
                ]
            )
        train_dataset = datasets.SVHN(dataset_dir, split="train", download=True, transform=transform_train)

        validation_dataset = datasets.SVHN(dataset_dir, split="test", download=True, transform=transform_test)
    else:
        train_dataset, validation_dataset = get_dataset(
            configs.dataset.name,
            configs.dataset.img_height,
            configs.dataset.img_width,
            dataset_dir=configs.dataset.root,
            transform=configs.dataset.transform,
        )

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=configs.run.batch_size,
        shuffle=int(configs.dataset.shuffle),
        pin_memory=True,
        num_workers=configs.dataset.num_workers,
    )
    validation_loader = torch.utils.data.DataLoader(
        dataset=validation_dataset,
        batch_size=configs.run.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=configs.dataset.num_workers,
    )
    return train_loader, validation_loader


def make_model(device: Device, random_state: int = None) -> nn.Module:
    if "mlp" in configs.model.name.lower():
        model = eval(configs.model.name)(
            n_feat=configs.dataset.img_height * configs.dataset.img_width,
            n_class=configs.dataset.n_class,
            hidden_list=configs.model.hidden_list,
            block_list=configs.model.block_list,
            in_bit=configs.quantize.input_bit,
            w_bit=configs.quantize.weight_bit,
            mode=configs.model.mode,
            v_max=configs.quantize.v_max,
            v_pi=configs.quantize.v_pi,
            act_thres=configs.model.act_thres,
            photodetect=False,
            bias=False,
            device=device,
        ).to(device)
        model.reset_parameters(random_state, morr_init=int(configs.morr.morr_init))
    elif "cnn" in configs.model.name.lower():
        model = eval(configs.model.name)(
            img_height=configs.dataset.img_height,
            img_width=configs.dataset.img_width,
            in_channels=configs.dataset.in_channel,
            num_classes=configs.dataset.n_class,
            kernel_list=configs.model.kernel_list,
            kernel_size_list=configs.model.kernel_size_list,
            pool_out_size=configs.model.pool_out_size,
            stride_list=configs.model.stride_list,
            padding_list=configs.model.padding_list,
            hidden_list=configs.model.hidden_list,
            block_list=configs.model.block_list,
            in_bit=configs.quantize.input_bit,
            w_bit=configs.quantize.weight_bit,
            mode=configs.model.mode,
            v_max=configs.quantize.v_max,
            v_pi=configs.quantize.v_pi,
            act_thres=configs.model.act_thres,
            photodetect=False,
            bias=False,
            # morr configuartion
            MORRConfig=eval(configs.morr.config),
            trainable_morr_bias=configs.morr.trainable_bias,
            trainable_morr_scale=configs.morr.trainable_scale,
            device=device,
        ).to(device)
        model.reset_parameters(random_state, morr_init=int(configs.morr.morr_init))
    else:
        model = None
        raise NotImplementedError(f"Not supported model name: {configs.model.name}")

    return model


def make_optimizer(model: nn.Module) -> Optimizer:
    if configs.optimizer.name == "sgd":
        optimizer = torch.optim.SGD(
            (p for p in model.parameters() if p.requires_grad),
            lr=configs.optimizer.lr,
            momentum=configs.optimizer.momentum,
            weight_decay=configs.optimizer.weight_decay,
            nesterov=True,
        )
    elif configs.optimizer.name == "adam":
        optimizer = torch.optim.Adam(
            (p for p in model.parameters() if p.requires_grad),
            lr=configs.optimizer.lr,
            weight_decay=configs.optimizer.weight_decay,
        )
    elif configs.optimizer.name == "adamw":
        optimizer = torch.optim.AdamW(
            (p for p in model.parameters() if p.requires_grad),
            lr=configs.optimizer.lr,
            weight_decay=configs.optimizer.weight_decay,
        )
    else:
        raise NotImplementedError(configs.optimizer.name)

    return optimizer


def make_scheduler(optimizer: Optimizer) -> Scheduler:
    if configs.scheduler.name == "constant":
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1)
    elif configs.scheduler.name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=configs.run.n_epochs, eta_min=configs.scheduler.lr_min
        )
    elif configs.scheduler.name == "exp":
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=configs.scheduler.lr_gamma)
    else:
        raise NotImplementedError(configs.scheduler.name)

    return scheduler


def make_criterion() -> nn.Module:
    if configs.criterion.name == "nll":
        criterion = nn.NLLLoss()
    elif configs.criterion.name == "mse":
        criterion = nn.MSELoss()
    elif configs.criterion.name == "ce":
        criterion = nn.CrossEntropyLoss()
    else:
        raise NotImplementedError(configs.criterion.name)
    return criterion
