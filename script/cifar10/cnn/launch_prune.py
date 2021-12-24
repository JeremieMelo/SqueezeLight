"""
Description:
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2021-12-20 16:43:20
LastEditors: Jiaqi Gu (jqgu@utexas.edu)
LastEditTime: 2021-12-23 23:37:28
"""

import os
import subprocess
from multiprocessing import Pool

import mlflow
from pyutils.general import ensure_dir, logger
from torchpack.utils.config import configs

dataset = "cifar10"
model = "cnn"
exp = "prune"
root = f"log/{dataset}/{model}/{exp}"
script = "train.py"
config_file = f"config/{dataset}/{model}/train/{exp}.yml"
configs.load(config_file, recursive=True)


def task_launcher(args):
    pres = ["python3", script, config_file]
    wbit, inbit, id = args
    with open(os.path.join(root, f"wb-{wbit}_inbit-{inbit}_run-{id}.log"), "w") as wfid:
        exp = [
            f"--run.random_state={41+id}",
            f"--quantize.weight_bit={wbit}",
            f"--quantize.input_bit={inbit}",
            f"--checkpoint.resume=1",
            f"--checkpoint.model_comment=topk-4",
        ]
        logger.info(f"running command {' '.join(pres + exp)}")
        subprocess.call(pres + exp, stderr=wfid, stdout=wfid)


if __name__ == "__main__":
    ensure_dir(root)
    mlflow.set_experiment(configs.run.experiment)  # set experiments first

    tasks = [(8, 8, 1)]

    with Pool(1) as p:
        p.map(task_launcher, tasks)
    logger.info(f"Exp: {configs.run.experiment} Done.")
