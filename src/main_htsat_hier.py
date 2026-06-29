from datetime import datetime
from pathlib import Path
from time import time
import os
import sys

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = Path(__file__).resolve().parent
os.chdir(ROOT)
sys.path.append(str(ROOT))
sys.path.append(str(CODE_DIR))

from detect_any_sound_htsat_hier import HierDASM_HTSAT
from recipes.audioset_strong.detect_any_sound.htsat.lr_set import get_lr_htsat
from recipes.audioset_strong.detect_any_sound.htsat.open_vocabulary import OV_DASM_HTSAT_Trainer
from recipes.audioset_strong.setting import (
    BestModels,
    dataset_setting,
    get_encoder,
    optimizer_and_scheduler_setting,
    prepare_run,
)
from runtime_patches import apply_runtime_patches, configure_runtime
from train_htsat_hier import HierDASM_HTSAT_Trainer


if __name__ == "__main__":
    configure_runtime(disable_cudnn=True)
    apply_runtime_patches()
    configs, my_logger, args = prepare_run()
    net = HierDASM_HTSAT(**configs["DASM_HTSAT"]["init_kwargs"])
    total_lr = get_lr_htsat(net, configs, my_logger.logger)
    encoder = get_encoder(configs)
    train_loader, val_loader, test_loader = dataset_setting(encoder, configs, my_logger.logger)
    optimizer, scheduler = optimizer_and_scheduler_setting(total_lr, train_loader, configs, my_logger.logger)

    if args.multigpu:
        net = nn.DataParallel(net)
    net = net.to(configs["training"]["device"])

    trainer_class = HierDASM_HTSAT_Trainer if not args.open_vocabulary else OV_DASM_HTSAT_Trainer
    trainer = trainer_class(
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

    if not configs["generals"]["test_only"]:
        my_logger.logger.info("   training starts!")
        start_time = time()
        bestmodels = BestModels()

        init_from_path = configs["training"].get("init_from_path")

        if ("finetune_mlm" in configs["generals"].keys()) and configs["generals"]["finetune_mlm"]:
            params_dict = torch.load(init_from_path or configs["training"]["best_paths"])
            params = {
                k: v
                for k, v in params_dict.items()
                if all(keyword not in k for keyword in ["at_query", "query_projector", "at_head.", "sed_head."])
            }
            trainer.net.load_state_dict(params, strict=False)
        elif configs["generals"]["load_from_existed_path"]:
            if init_from_path is None:
                raise RuntimeError("training.init_from_path must be set when load_from_existed_path is true")
            trainer.net.load_state_dict(torch.load(init_from_path), strict=False)
            if not configs["training"].get("skip_initial_validation", True):
                val_metrics = trainer.validation(epoch=-1)
                bestmodels.update(trainer.net, 0, my_logger.logger, val_metrics)
                bestmodels.save_bests(configs["training"]["best_paths"])

        validation_interval = max(1, int(configs["generals"].get("validation_interval", 1)))
        total_epochs = configs["training"]["scheduler"]["n_epochs"]
        for epoch in range(total_epochs):
            trainer.train(epoch)
            should_validate = ((epoch + 1) % validation_interval == 0) or (epoch + 1 == total_epochs)
            if should_validate:
                val_metrics = trainer.validation(epoch)
                bestmodels.update(trainer.net, epoch + 1, my_logger.logger, val_metrics)
                bestmodels.save_bests(configs["training"]["best_paths"])

        my_logger.logger.info(
            "        best student/teacher val_metrics: %.3f"
            % bestmodels.save_bests(configs["training"]["best_paths"])
        )
        my_logger.logger.info("   training took %.2f mins" % ((time() - start_time) / 60))

    my_logger.logger.info("   test starts!")
    trainer.net.load_state_dict(torch.load(configs["training"]["best_paths"]), strict=False)
    trainer.test()
    my_logger.logger.info("date & time of end is : " + str(datetime.now()).split(".")[0])
    print("<" * 30 + "DONE!" + ">" * 30)
    my_logger.close()
