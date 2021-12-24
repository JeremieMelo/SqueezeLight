'''
Description:
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2021-05-10 20:34:02
LastEditors: Jiaqi Gu (jqgu@utexas.edu)
LastEditTime: 2021-12-23 23:56:38
'''
#!/usr/bin/env python
# coding=UTF-8
import argparse
import os
from typing import Iterable

import mlflow
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pyutils.config import configs

from core import builder
from pyutils.general import logger as lg
from pyutils.loss import KLLossMixed
from pyutils.torch_train import (BestKModelSaver, count_parameters,
                                 get_learning_rate, load_model,
                                 set_torch_deterministic)
from pyutils.typing import Criterion, DataLoader, Optimizer, Scheduler


def train(
        model: nn.Module,
        train_loader: DataLoader,
        optimizer: Optimizer,
        scheduler: Scheduler,
        epoch: int,
        criterion: Criterion,
        prune_finegrain: bool,
        device: torch.device) -> None:
    model.train()
    step = epoch * len(train_loader)
    correct = 0

    if(prune_finegrain):
        drop_masks = model.get_finegrain_drop_mask(topk=int(configs.prune.topk))
        n_drop = {layer: torch.sum(1-mask.float()).cpu().data.item() for layer, mask in drop_masks.items()}
        drop_rates = {layer: n_drop[layer]/mask.numel() for layer, mask in drop_masks.items()}
        lg.info(f"Finegrain drop mask {n_drop} drop rate {drop_rates}")

    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        optimizer.zero_grad()

        output = model(data)
        classify_loss = criterion(output, target)

        pred = output.data.max(1)[1]
        correct += pred.eq(target.data).cpu().sum()

        loss = classify_loss

        loss.backward()

        optimizer.step()
        step += 1

        if batch_idx % int(configs.run.log_interval) == 0:
            lg.info('Train Epoch: {} [{:7d}/{:7d} ({:3.0f}%)] Loss: {:.4f} Class Loss: {:.4f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data.item(), classify_loss.data.item()))
            mlflow.log_metrics({"train_loss": loss.item()}, step=step)

    scheduler.step()
    accuracy = 100. * correct.float() / len(train_loader.dataset)
    lg.info(
        f"Train Accuracy: {correct}/{len(train_loader.dataset)} ({accuracy:.2f})%")
    mlflow.log_metrics({"train_acc": accuracy.item(),
                        "lr": get_learning_rate(optimizer)}, step=epoch)

def validate(
        model: nn.Module,
        validation_loader: DataLoader,
        epoch: int,
        criterion: Criterion,
        loss_vector: Iterable,
        accuracy_vector: Iterable,
        prune_finegrain: bool,
        device: torch.device) -> None:
    model.eval()
    val_loss, correct = 0, 0
    with torch.no_grad():
        for data, target in validation_loader:
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            output = model(data)

            val_loss += criterion(output, target).data.item()
            pred = output.data.max(1)[1]
            correct += pred.eq(target.data).cpu().sum()

    val_loss /= len(validation_loader)
    loss_vector.append(val_loss)

    accuracy = 100. * correct.float() / len(validation_loader.dataset)
    accuracy_vector.append(accuracy)

    lg.info('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        val_loss, correct, len(validation_loader.dataset), accuracy))
    mlflow.log_metrics({"val_acc": accuracy.data.item(),
                        "val_loss": val_loss}, step=epoch)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('config', metavar='FILE', help='config file')
    # parser.add_argument('--run-dir', metavar='DIR', help='run directory')
    # parser.add_argument('--pdb', action='store_true', help='pdb')
    args, opts = parser.parse_known_args()

    configs.load(args.config, recursive=True)
    configs.update(opts)

    if (torch.cuda.is_available() and int(configs.run.use_cuda)):
        torch.cuda.set_device(configs.run.gpu_id)
        device = torch.device('cuda:'+str(configs.run.gpu_id))
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device('cpu')
        torch.backends.cudnn.benchmark = False

    if(int(configs.run.deterministic) == True):
        set_torch_deterministic()

    model = builder.make_model(device, int(configs.run.random_state) if int(
        configs.run.deterministic) else None)

    train_loader, validation_loader = builder.make_dataloader()
    optimizer = builder.make_optimizer(model)
    scheduler = builder.make_scheduler(optimizer)
    criterion = builder.make_criterion().to(device)
    saver = BestKModelSaver(k=int(configs.checkpoint.save_best_model_k))

    lg.info(f'Number of parameters: {count_parameters(model)}')

    model_name = f"{configs.model.name}_wb-{configs.quantize.weight_bit}_ib-{configs.quantize.input_bit}"
    checkpoint = f"./checkpoint/{configs.checkpoint.checkpoint_dir}/{model_name}_{configs.checkpoint.model_comment}.pt"

    lg.info(f"Current checkpoint: {checkpoint}")

    mlflow.set_experiment(configs.run.experiment)
    experiment = mlflow.get_experiment_by_name(configs.run.experiment)

    # run_id_prefix = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    mlflow.start_run(run_name=model_name)
    mlflow.log_params({
        "exp_name": configs.run.experiment,
        "exp_id": experiment.experiment_id,
        "run_id": mlflow.active_run().info.run_id,
        "inbit": configs.quantize.input_bit,
        "wbit": configs.quantize.weight_bit,
        "init_lr": configs.optimizer.lr,
        "checkpoint": checkpoint,
        "restore_checkpoint": configs.checkpoint.restore_checkpoint,
        "pid": os.getpid()
    })

    lossv, accv = [0], [0]
    epoch = 0
    try:
        lg.info(
            f"Experiment {configs.run.experiment} ({experiment.experiment_id}) starts. Run ID: ({mlflow.active_run().info.run_id}). PID: ({os.getpid()}). PPID: ({os.getppid()}). Host: ({os.uname()[1]})")
        lg.info(configs)
        prune_finegrain = False
        if int(configs.checkpoint.resume) and len(configs.checkpoint.restore_checkpoint) > 0:
            load_model(model, configs.checkpoint.restore_checkpoint,
                       ignore_size_mismatch=int(configs.checkpoint.no_linear))

            lg.info("Validate resumed model...")
            validate(
                model,
                validation_loader,
                0,
                criterion,
                lossv,
                accv,
                False,
                device)
            if int(configs.prune.topk) > 0:
                model.get_finegrain_drop_mask(topk=int(configs.prune.topk))
                prune_finegrain = True

        for epoch in range(1, int(configs.run.n_epochs)+1):
            train(
                model,
                train_loader,
                optimizer,
                scheduler,
                epoch,
                criterion,
                prune_finegrain,
                device)
            validate(
                model,
                validation_loader,
                epoch,
                criterion,
                lossv,
                accv,
                prune_finegrain,
                device)
            saver.save_model(
                model,
                accv[-1],
                epoch=epoch,
                path=checkpoint,
                save_model=False,
                print_msg=True
            )
    except KeyboardInterrupt:
        lg.warning("Ctrl-C Stopped")


if __name__ == "__main__":
    main()
