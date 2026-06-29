from datetime import datetime
from pathlib import Path
from time import time
import argparse
import os
import shutil
import sys

import torch
import torch.nn as nn
import yaml

ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = Path(__file__).resolve().parent
os.chdir(ROOT)
sys.path.append(str(ROOT))
sys.path.append(str(CODE_DIR))

from detect_any_sound_htsat_hier import HierDASM_HTSAT
from recipes.audioset_strong.detect_any_sound.htsat.lr_set import get_lr_htsat
from recipes.audioset_strong.setting import (
    dataset_setting,
    get_encoder,
    get_save_directories,
    optimizer_and_scheduler_setting,
    prepare_run,
)
from runtime_patches import apply_runtime_patches, configure_runtime
from train_htsat_hier import HierDASM_HTSAT_Trainer


class _NoOpTensorboardWriter:
    def add_scalar(self, *args, **kwargs):
        return None


if __name__ == "__main__":
    configure_runtime(disable_cudnn=True)
    apply_runtime_patches()

    configs, my_logger, args = prepare_run()
    if getattr(my_logger, "tensorboard_writer", None) is None:
        my_logger.tensorboard_writer = _NoOpTensorboardWriter()
    start_time = time()

    # Build the validation loader and run validation only. The normal test-only
    # path writes per-file sed_scores to disk and is much slower for val outputs.
    configs["generals"]["test_only"] = False
    configs["generals"]["save_val_pred_tsv"] = True
    configs["generals"]["save_pred_tsv"] = False
    configs["generals"]["test_on_public_eval"] = False

    net = HierDASM_HTSAT(**configs["DASM_HTSAT"]["init_kwargs"])
    total_lr = get_lr_htsat(net, configs, my_logger.logger)
    encoder = get_encoder(configs)
    train_loader, val_loader, test_loader = dataset_setting(encoder, configs, my_logger.logger)
    optimizer, scheduler = optimizer_and_scheduler_setting(total_lr, train_loader, configs, my_logger.logger)

    if args.multigpu:
        net = nn.DataParallel(net)
    net = net.to(configs["training"]["device"])

    trainer = HierDASM_HTSAT_Trainer(
        optimizer=optimizer,
        my_logger=my_logger,
        net=net,
        config=configs,
        encoder=encoder,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=configs["training"]["device"],
    )

    model_path = configs["training"].get("init_from_path") or configs["training"]["best_paths"]
    my_logger.logger.info("   loading model from: " + model_path)
    trainer.net.load_state_dict(torch.load(model_path), strict=False)

    my_logger.logger.info("   validation prediction starts!")
    trainer.validation(epoch=4)

    src_pred = os.path.join(configs["generals"]["save_folder"], "val_pred_events_epoch5.tsv")
    dst_pred = os.path.join(configs["generals"]["save_folder"], "val_pred_events_best.tsv")
    if os.path.exists(src_pred):
        shutil.copyfile(src_pred, dst_pred)
        my_logger.logger.info("   copied prediction TSV to: " + dst_pred)

    my_logger.logger.info("   validation prediction took %.2f mins" % ((time() - start_time) / 60))
    my_logger.logger.info("date & time of end is : " + str(datetime.now()).split(".")[0])
    print("<" * 30 + "VAL PRED DONE!" + ">" * 30)
    my_logger.close()
