from pathlib import Path
import argparse
import json
import logging
import os
import sys
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
import time

import pandas as pd
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = Path(__file__).resolve().parent
os.chdir(ROOT)
sys.path.append(str(ROOT))
sys.path.append(str(CODE_DIR))

from recipes.audioset_strong.setting import (
    dataset_setting,
    get_encoder,
    get_save_directories,
    setup_seed,
)
from recipes.desed.setting import get_logger
from src.codec.decoder import batched_decode_preds
from src.evaluation_measures import compute_psds_from_scores
from src.models.detect_any_sound.detect_any_sound_htast import DASM_HTSAT
from src.utils import load_yaml_with_relative_ref
from train_htsat_quantile_mf import EventSpecificMedianFilterHTSATTrainer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(CODE_DIR / "config_htsat_event_mf.yaml"),
    )
    parser.add_argument(
        "--save_folder",
        default=str(ROOT / "innovation/event_specific_median_filtering_clean/runs/as_full_val_audio_query_event_mf_psds_20260414"),
    )
    parser.add_argument(
        "--model_path",
        default=str(ROOT / "weights/ocl_caw_best_model.pt"),
    )
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--psds_num_jobs", type=int, default=1)
    parser.add_argument("--psds_heartbeat_sec", type=int, default=30)
    return parser.parse_args()


def append_threshold_events(score_dfs, encoder, threshold, out_path):
    rows = []
    for audio_id, score_df in score_dfs.items():
        filename = f"{audio_id}.wav"
        event_classes = [label for label in encoder.labels if label in score_df.columns]
        scores = score_df[event_classes].to_numpy()
        active = scores > threshold
        for class_idx, label in enumerate(event_classes):
            change_indices = encoder.find_contiguous_regions(active[:, class_idx])
            for onset_frame, offset_frame in change_indices:
                rows.append({
                    "filename": filename,
                    "onset": encoder._frame_to_time(onset_frame),
                    "offset": encoder._frame_to_time(offset_frame),
                    "event_label": label,
                })
    if rows:
        pd.DataFrame(rows).to_csv(out_path, sep="\t", index=False, header=False, mode="a")


def mean_psds_per_known_type(psds_single, type_dict):
    grouped = OrderedDict([
        ("common", []),
        ("rare", []),
    ])
    missing = []
    for event, value in psds_single.items():
        event_type = type_dict.get(event)
        if event_type in grouped:
            grouped[event_type].append(float(value))
        else:
            missing.append(event)
    means = {
        key: round(sum(values) / len(values), 4) if values else 0.0
        for key, values in grouped.items()
    }
    return means, sorted(missing)


def compute_psds_with_heartbeat(scores, val_tsv, val_dur, num_jobs, heartbeat_sec):
    def _compute():
        return compute_psds_from_scores(
            scores,
            val_tsv,
            val_dur,
            dtc_threshold=0.7,
            gtc_threshold=0.7,
            cttc_threshold=None,
            alpha_ct=0,
            alpha_st=0,
            max_efpr=100,
            num_jobs=num_jobs,
            save_dir=None,
        )

    start = time.time()
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_compute)
        while not future.done():
            elapsed = int(time.time() - start)
            print(f"PSDS still running... elapsed={elapsed}s", flush=True)
            time.sleep(max(1, heartbeat_sec))
        return future.result()


def main():
    args = parse_args()
    configs = load_yaml_with_relative_ref(args.config)
    configs["generals"]["savepsds"] = True
    configs["generals"]["save_pred_tsv"] = True
    configs = get_save_directories(configs, args.save_folder)

    my_logger = get_logger(
        configs["generals"]["save_folder"],
        need_tensorboard=False,
        log_level=getattr(logging, configs["generals"]["log_level"].upper()),
    )
    my_logger.logger.info("Starting val prediction + PSDS with event-specific median filter")

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    configs["training"]["device"] = device
    my_logger.logger.info(f"Device: {device}")

    setup_seed(configs["training"]["seed"])
    configs["generals"]["test_only"] = True
    configs["dataset"]["test_tsv"] = configs["dataset"]["val_tsv"]
    configs["dataset"]["test_dur"] = configs["dataset"]["val_dur"]
    configs["dataset"]["test_folder"] = configs["dataset"]["val_folder"]

    net = DASM_HTSAT(**configs["DASM_HTSAT"]["init_kwargs"])
    net = net.to(device)
    encoder = get_encoder(configs)
    train_loader, val_loader, test_loader = dataset_setting(encoder, configs, my_logger.logger)

    trainer = EventSpecificMedianFilterHTSATTrainer(
        optimizer=None,
        my_logger=my_logger,
        net=net,
        config=configs,
        encoder=encoder,
        scheduler=None,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
    )

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(args.model_path)
    my_logger.logger.info(f"Loading model from {args.model_path}")
    trainer.net.load_state_dict(torch.load(args.model_path, map_location=device))
    trainer.net.eval()

    save_folder = Path(configs["generals"]["save_folder"])
    pred_path = save_folder / "val_pred_events.tsv"
    with open(pred_path, "w", encoding="utf-8") as f:
        f.write("filename\tonset\toffset\tevent_label\n")

    val_tsv = configs["dataset"]["val_tsv"]
    val_dur = configs["dataset"]["val_dur"]
    val_df = pd.read_csv(val_tsv, sep="\t")
    val_events_set = set(val_df["event_label"])
    extra_events = set(encoder.labels) - val_events_set
    filter_type = configs["training"].get("filter_type", "median")
    val_scores_postprocessed_buffer = {}

    with torch.no_grad():
        for batch in tqdm(test_loader, total=len(test_loader), leave=False, desc="val prediction+psds"):
            wav, labels, pad_mask, idx, filename, path = batch
            wav = wav.to(device)
            feat = trainer.preprocess_eval(wav)
            pred = trainer.net(
                input=feat,
                pad_mask=pad_mask,
                **configs[trainer.net.get_model_name()]["val_kwargs"],
            )
            pred = trainer.Pred(*pred)

            raw_scores, scores = batched_decode_preds(
                strong_preds=pred.strong,
                filenames=path,
                encoder=encoder,
                filter=trainer.median_fiter,
                weak_preds=None,
                need_weak_mask=False,
                filter_type=filter_type,
            )
            if extra_events:
                for audio_id, score_df in scores.items():
                    scores[audio_id] = score_df.drop(extra_events, axis=1)
            val_scores_postprocessed_buffer.update(scores)
            append_threshold_events(scores, encoder, args.threshold, pred_path)

    my_logger.logger.info("Finished val prediction; calculating PSDS from in-memory scores")
    print(
        "Finished val prediction; calculating PSDS from in-memory scores "
        f"(num_clips={len(val_scores_postprocessed_buffer)}, num_jobs={args.psds_num_jobs})",
        flush=True,
    )
    psds, psds_single = compute_psds_with_heartbeat(
        val_scores_postprocessed_buffer,
        val_tsv,
        val_dur,
        num_jobs=args.psds_num_jobs,
        heartbeat_sec=args.psds_heartbeat_sec,
    )
    partial_psds, missing_type_events = mean_psds_per_known_type(psds_single, trainer.type_dict)
    summary = OrderedDict([
        ("psds", float(psds)),
        ("psds_common", float(partial_psds["common"])),
        ("psds_rare", float(partial_psds["rare"])),
        ("n_events_missing_common_rare_type", len(missing_type_events)),
        ("events_missing_common_rare_type", missing_type_events),
    ])
    psds_single_rounded = OrderedDict(
        sorted(((k, round(float(v), 4)) for k, v in psds_single.items()), key=lambda x: x[1])
    )
    with open(save_folder / "single_psds.json", "w", encoding="utf-8") as f:
        json.dump(psds_single_rounded, f, indent=4)
    with open(save_folder / "val_psds_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)

    my_logger.logger.info(
        "PSDS summary: psds=%.5f, common=%.5f, rare=%.5f"
        % (summary["psds"], summary["psds_common"], summary["psds_rare"])
    )
    my_logger.logger.info(f"Saved val predictions to {pred_path}")
    my_logger.close()
    print(pred_path)
    print(save_folder / "val_psds_summary.json")


if __name__ == "__main__":
    main()
