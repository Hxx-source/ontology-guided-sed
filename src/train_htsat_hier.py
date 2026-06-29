from pathlib import Path
import os
import sys
import json
import inspect
from collections import OrderedDict

import pandas as pd
import sed_scores_eval
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
import numpy as np
from sed_scores_eval.base_modules.scores import create_score_dataframe

ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = Path(__file__).resolve().parent
os.chdir(ROOT)
sys.path.append(str(ROOT))
sys.path.append(str(CODE_DIR))

from hierarchy_consistency import HierarchyConsistencyHelper
from recipes.audioset_strong.detect_any_sound.htsat.train import DASM_HTSAT_Trainer
from recipes.audioset_strong.detect_any_sound.passt.train import multi_class_to_multi_label, multi_label_to_multi_class
from src.codec.decoder import batched_decode_preds, decode_pred_batch_fast
from src.codec.encoder import Encoder
from src.evaluation_measures import event_based_evaluation_df, segment_based_evaluation_df


class HierDASM_HTSAT_Trainer(DASM_HTSAT_Trainer):

    @staticmethod
    def _load_label_subset(labels, path, label_column="event_label"):
        if not path:
            return None, None

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Target label file does not exist: {path}")

        if path.suffix.lower() in (".csv", ".tsv"):
            sep = "\t" if path.suffix.lower() == ".tsv" else ","
            df = pd.read_csv(path, sep=sep)
            if label_column not in df.columns:
                raise ValueError(f"{path} does not contain column: {label_column}")
            target_labels = [str(label).strip() for label in df[label_column].dropna()]
        elif path.suffix.lower() in (".xls", ".xlsx"):
            df = pd.read_excel(path)
            if label_column not in df.columns:
                raise ValueError(f"{path} does not contain column: {label_column}")
            target_labels = [str(label).strip() for label in df[label_column].dropna()]
        else:
            target_labels = [
                line.strip()
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

        target_labels = list(dict.fromkeys(target_labels))
        label_to_idx = {label: idx for idx, label in enumerate(labels)}
        unknown = [label for label in target_labels if label not in label_to_idx]
        if unknown:
            raise ValueError(f"Unknown target labels in {path}: {unknown}")
        target_indices = torch.tensor([label_to_idx[label] for label in target_labels], dtype=torch.long)
        return target_labels, target_indices

    @staticmethod
    def _load_target_pairs(labels, path, child_column="child_label", parent_column="parent_label"):
        if not path:
            return None

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Target pair file does not exist: {path}")

        if path.suffix.lower() in (".xls", ".xlsx"):
            df = pd.read_excel(path)
        else:
            sep = "\t" if path.suffix.lower() == ".tsv" else ","
            df = pd.read_csv(path, sep=sep)
        missing = [column for column in (child_column, parent_column) if column not in df.columns]
        if missing:
            raise ValueError(f"{path} does not contain columns: {missing}")

        label_to_idx = {label: idx for idx, label in enumerate(labels)}
        pairs = OrderedDict()
        unknown = []
        for _, row in df[[child_column, parent_column]].dropna().iterrows():
            child_label = str(row[child_column]).strip()
            parent_label = str(row[parent_column]).strip()
            if child_label not in label_to_idx:
                unknown.append(child_label)
                continue
            if parent_label not in label_to_idx:
                unknown.append(parent_label)
                continue
            pairs.setdefault(child_label, [])
            if parent_label not in pairs[child_label]:
                pairs[child_label].append(parent_label)
        if unknown:
            raise ValueError(f"Unknown target pair labels in {path}: {sorted(set(unknown))}")
        return pairs

    @staticmethod
    def _classwise_f1_dataframe(event_classwise: dict, segment_classwise: dict):
        rows = []
        labels = sorted(set(event_classwise.keys()) | set(segment_classwise.keys()))
        for label in labels:
            event_entry = event_classwise.get(label, {})
            segment_entry = segment_classwise.get(label, {})
            rows.append({
                "event_label": label,
                "event_F1": event_entry.get("f_measure", {}).get("f_measure", 0.0),
                "event_precision": event_entry.get("f_measure", {}).get("precision", 0.0),
                "event_recall": event_entry.get("f_measure", {}).get("recall", 0.0),
                "segment_F1": segment_entry.get("f_measure", {}).get("f_measure", 0.0),
                "segment_precision": segment_entry.get("f_measure", {}).get("precision", 0.0),
                "segment_recall": segment_entry.get("f_measure", {}).get("recall", 0.0),
            })
        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values(["event_F1", "segment_F1", "event_label"], ascending=[True, True, True]).reset_index(drop=True)
        return df

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        hier_cfg = self.config.get("hierarchy_consistency", {})
        self.hierarchy_cfg = hier_cfg
        self.hierarchy_enabled = bool(hier_cfg.get("enabled", False))
        self.hierarchy_feature_loss_weight = float(
            hier_cfg.get("feature_loss_weight", hier_cfg.get("loss_weight", 0.08))
        )
        self.hierarchy_score_loss_weight = float(hier_cfg.get("score_loss_weight", 0.0))
        self.hierarchy_active_epochs = hier_cfg.get("active_epochs")
        self.eval_single_gpu = bool(hier_cfg.get("eval_single_gpu", True))
        train_cfg = self.config.get("training", {})
        self.target_loss_labels, self.target_loss_indices = self._load_label_subset(
            labels=self.encoder.labels,
            path=train_cfg.get("target_loss_labels_path"),
            label_column=train_cfg.get("target_loss_label_column", "event_label"),
        )
        if self.target_loss_labels is not None:
            self.my_logger.logger.info(
                "target supervised loss enabled for %d labels: %s"
                % (len(self.target_loss_labels), train_cfg.get("target_loss_labels_path"))
            )
        eval_target_path = train_cfg.get("eval_target_labels_path", train_cfg.get("target_loss_labels_path"))
        self.eval_target_only = bool(train_cfg.get("eval_target_only", self.target_loss_labels is not None))
        if self.eval_target_only:
            self.eval_target_labels, self.eval_target_indices = self._load_label_subset(
                labels=self.encoder.labels,
                path=eval_target_path,
                label_column=train_cfg.get("eval_target_label_column", train_cfg.get("target_loss_label_column", "event_label")),
            )
            if self.eval_target_labels is None:
                raise ValueError("training.eval_target_only is true, but no eval/target label file is configured")
            self.eval_encoder = Encoder(
                self.eval_target_labels,
                audio_len=self.config["feature"]["audio_max_len"],
                frame_len=self.config["feature"]["win_length"],
                frame_hop=self.config["feature"]["hopsize"],
                net_pooling=self.config["feature"]["net_subsample"],
                sr=self.config["feature"]["sr"],
            )
            self.my_logger.logger.info(
                "target-only evaluation enabled for %d labels: %s"
                % (len(self.eval_target_labels), eval_target_path)
            )
        else:
            self.eval_target_labels = None
            self.eval_target_indices = None
            self.eval_encoder = self.encoder
        self.hierarchy_helper = None
        if self.hierarchy_enabled:
            self.hierarchy_helper = HierarchyConsistencyHelper(
                labels=self.encoder.labels,
                ontology_path=hier_cfg["ontology_path"],
                class_stats_path=hier_cfg["class_stats_path"],
                tail_threshold=hier_cfg.get("tail_threshold"),
                distance_decay=hier_cfg.get("distance_decay", 0.35),
                min_weight=hier_cfg.get("min_class_weight", 0.25),
                max_weight=hier_cfg.get("max_class_weight", 4.0),
                require_positive_ancestor=hier_cfg.get("require_positive_ancestor", True),
                positive_margin=hier_cfg.get("positive_margin", 0.7),
                negative_margin=hier_cfg.get("negative_margin", 0.35),
                sibling_weight=hier_cfg.get("sibling_weight", 0.35),
                use_sibling_separation=hier_cfg.get("use_sibling_separation", True),
                max_ancestor_distance=hier_cfg.get("max_ancestor_distance", 1),
                score_margin=hier_cfg.get("score_margin", 0.0),
                score_source=hier_cfg.get("score_source", "both"),
                score_detach_ancestor=hier_cfg.get("score_detach_ancestor", False),
                target_labels=hier_cfg.get("target_labels"),
                target_labels_path=hier_cfg.get("target_labels_path"),
                target_label_column=hier_cfg.get("target_label_column", "event_label"),
                target_pairs=hier_cfg.get("target_pairs"),
                target_pairs_path=hier_cfg.get("target_pairs_path"),
                target_pair_child_column=hier_cfg.get("target_pair_child_column", "child_label"),
                target_pair_parent_column=hier_cfg.get("target_pair_parent_column", "parent_label"),
                target_weight_mode=hier_cfg.get("target_weight_mode", "class_weight"),
                adaptive_pair_weight_config=hier_cfg.get("adaptive_pair_weight_config"),
            )
            if self.hierarchy_helper.target_labels is not None:
                self.my_logger.logger.info(
                    "hierarchy consistency target labels: %d from %s"
                    % (len(self.hierarchy_helper.target_labels), hier_cfg.get("target_labels_path"))
                )
        query_mix_cfg = self.config.get("query_parent_mix", {})
        self.query_parent_mix_enabled = bool(query_mix_cfg.get("enabled", False))
        if self.query_parent_mix_enabled:
            pairs_by_child = self._load_target_pairs(
                labels=self.encoder.labels,
                path=query_mix_cfg.get("target_pairs_path"),
                child_column=query_mix_cfg.get("target_pair_child_column", "child_label"),
                parent_column=query_mix_cfg.get("target_pair_parent_column", "parent_label"),
            )
            pair_indices = []
            multi_parent_children = []
            label_to_idx = {label: idx for idx, label in enumerate(self.encoder.labels)}
            for child_label, parent_labels in (pairs_by_child or {}).items():
                parent_label = parent_labels[0]
                if len(parent_labels) > 1:
                    multi_parent_children.append((child_label, parent_labels))
                pair_indices.append((label_to_idx[child_label], label_to_idx[parent_label]))
            net_ref = self.net.module if isinstance(self.net, nn.DataParallel) else self.net
            if not hasattr(net_ref, "set_query_parent_mix"):
                raise AttributeError(f"Model {type(net_ref).__name__} does not support query_parent_mix")
            net_ref.set_query_parent_mix(
                pair_indices=pair_indices,
                alpha=query_mix_cfg.get("alpha", 0.1),
                mode=query_mix_cfg.get("mix_mode", "convex"),
            )
            if multi_parent_children:
                self.my_logger.logger.warning(
                    "query_parent_mix found multiple parents for %d children; using the first listed parent for each",
                    len(multi_parent_children),
                )
            self.my_logger.logger.info(
                "query_parent_mix enabled for %d child-parent pairs from %s (alpha=%.4f, mode=%s)"
                % (
                    len(pair_indices),
                    query_mix_cfg.get("target_pairs_path"),
                    float(query_mix_cfg.get("alpha", 0.1)),
                    str(query_mix_cfg.get("mix_mode", "convex")),
                )
            )

    @staticmethod
    def _odd_window(value: int):
        value = max(1, int(value))
        return value + 1 if value % 2 == 0 else value

    @staticmethod
    def _esp_duration(durations: np.ndarray):
        durations = np.sort(durations.astype(float))
        if durations.size == 0:
            return None
        if durations.size == 1 or np.isclose(durations.max(), durations.min()):
            return float(durations.mean())
        x = (durations - durations.min()) / (durations.max() - durations.min())
        y = np.arange(1, durations.size + 1, dtype=float) / durations.size
        inflection_idx = int(np.argmax(y - x)) + 1
        return float(durations[:inflection_idx].mean())

    def _at_scores_for_hierarchy(self, pred):
        at_out = pred.other_dict.get("at_out")
        if at_out is None:
            return pred.weak

        out_type = self.config[self.net.get_model_name()]["init_kwargs"]["at_param"]["out_type"]
        if out_type == "logit":
            return multi_class_to_multi_label(F.softmax(at_out, dim=-1))
        return at_out

    def _target_indices_for_device(self, device):
        if self.target_loss_indices is None:
            return None
        return self.target_loss_indices.to(device)

    def _eval_indices_for_device(self, device):
        if self.eval_target_indices is None:
            return None
        return self.eval_target_indices.to(device)

    def _select_eval_tensor(self, tensor):
        eval_indices = self._eval_indices_for_device(tensor.device)
        if eval_indices is None:
            return tensor
        return tensor.index_select(1, eval_indices)

    def _eval_median_filter(self):
        median_filter = self.median_fiter
        if self.eval_target_indices is None or isinstance(median_filter, int):
            return median_filter
        return [median_filter[int(idx)] for idx in self.eval_target_indices.tolist()]

    @property
    def eval_events_set(self):
        if self.eval_target_labels is not None:
            return set(self.eval_target_labels)
        return self.val_events_set

    @staticmethod
    def _audio_id(filename):
        return Path(str(filename)).stem

    def _filtered_eval_files(self, tsv_path, dur_path, split_name):
        if self.eval_target_labels is None:
            return tsv_path, dur_path, None

        gt_path = os.path.join(
            self.config["generals"]["save_folder"],
            f"{split_name}_target30_ground_truth.tsv",
        )
        dur_out_path = os.path.join(
            self.config["generals"]["save_folder"],
            f"{split_name}_target30_durations.tsv",
        )
        os.makedirs(os.path.dirname(gt_path), exist_ok=True)
        if not os.path.exists(gt_path):
            gt_df = pd.read_csv(tsv_path, sep="\t")
            gt_df = gt_df[gt_df["event_label"].isin(self.eval_target_labels)].copy()
            gt_df.to_csv(gt_path, sep="\t", index=False)
        else:
            gt_df = pd.read_csv(gt_path, sep="\t")

        audio_ids = {self._audio_id(filename) for filename in gt_df["filename"].dropna().unique()}
        if not os.path.exists(dur_out_path):
            dur_df = pd.read_csv(dur_path, sep="\t")
            dur_df = dur_df[dur_df["filename"].map(self._audio_id).isin(audio_ids)].copy()
            dur_df.to_csv(dur_out_path, sep="\t", index=False)
        return gt_path, dur_out_path, audio_ids

    def _zero_score_dataframe(self, duration):
        duration = float(duration)
        timestamps = np.array([0.0, max(duration, 1e-6)], dtype=float)
        scores = np.zeros((1, len(self.eval_encoder.labels)), dtype=np.float32)
        return create_score_dataframe(scores, timestamps, self.eval_encoder.labels)

    def _align_scores_to_audio_ids(self, scores_buffer, audio_ids, dur_path):
        if audio_ids is None:
            return scores_buffer

        aligned = {
            audio_id: df
            for audio_id, df in scores_buffer.items()
            if audio_id in audio_ids
        }
        missing = audio_ids - set(aligned)
        if missing:
            dur_df = pd.read_csv(dur_path, sep="\t")
            duration_by_id = {
                self._audio_id(row["filename"]): row["duration"]
                for _, row in dur_df.iterrows()
            }
            for audio_id in sorted(missing):
                aligned[audio_id] = self._zero_score_dataframe(duration_by_id.get(audio_id, self.config["feature"]["audio_max_len"]))
        return aligned

    def _filter_pred_df_to_audio_ids(self, pred_df, audio_ids):
        if audio_ids is None or pred_df.empty:
            return pred_df
        return pred_df[pred_df["filename"].map(self._audio_id).isin(audio_ids)].copy()

    def _remove_extra_events(self, scores_buffer: dict, events):
        for audio_id, df in scores_buffer.items():
            scores_buffer[audio_id] = df.drop(events, axis=1, errors="ignore")
        return scores_buffer

    @property
    def median_fiter(self):
        if hasattr(self, "_median_filter"):
            return self._median_filter

        cfg = self.config["training"].get("classwise_median_filter", {})
        if not cfg.get("enabled", False):
            return super().median_fiter

        precomputed_path = cfg.get("precomputed_path")
        if precomputed_path:
            df = pd.read_csv(precomputed_path)
            if "event_label" not in df.columns:
                raise ValueError(f"{precomputed_path} does not contain column: event_label")
            window_column = cfg.get("window_column")
            if window_column is None:
                if "median_window" in df.columns:
                    window_column = "median_window"
                elif "median_window_frames" in df.columns:
                    window_column = "median_window_frames"
                else:
                    raise ValueError(
                        f"{precomputed_path} must contain median_window or median_window_frames"
                    )
            if window_column not in df.columns:
                raise ValueError(f"{precomputed_path} does not contain column: {window_column}")
            label_to_window = {
                str(row["event_label"]): self._odd_window(int(row[window_column]))
                for _, row in df.iterrows()
            }
            missing = [label for label in self.encoder.labels if label not in label_to_window]
            if missing:
                raise ValueError(f"Missing classwise median windows in {precomputed_path}: {missing}")
            self._median_filter = [label_to_window[label] for label in self.encoder.labels]
            out_path = cfg.get("save_path")
            if out_path:
                rows = [
                    {"event_label": label, "median_window": int(label_to_window[label])}
                    for label in self.encoder.labels
                ]
                pd.DataFrame(rows).to_csv(out_path, index=False)
            self.my_logger.logger.info(
                "classwise median filter loaded: min=%d, max=%d, mean=%.2f, source=%s"
                % (
                    min(self._median_filter),
                    max(self._median_filter),
                    float(np.mean(self._median_filter)),
                    precomputed_path,
                )
            )
            return self._median_filter

        pred_len = int(self.config["feature"]["pred_len"])
        audio_len = float(self.config["feature"]["audio_max_len"])
        default_window = self._odd_window(int(self.config["training"]["median_window"] / 156 * pred_len))
        min_window = self._odd_window(int(cfg.get("min_window", 1)))
        max_window = self._odd_window(int(cfg.get("max_window", default_window)))
        scaling_factor = float(cfg.get("scaling_factor", 1.0 / 3.0))
        statistic = cfg.get("statistic", "median")
        source_tsv = cfg.get("source_tsv") or self.config["dataset"]["strong_tsv"]

        df = pd.read_csv(source_tsv, sep="\t")
        df = df[df["event_label"].isin(self.encoder.labels)].copy()
        df["duration"] = (df["offset"] - df["onset"]).clip(lower=0)

        windows = []
        rows = []
        seconds_to_frames = pred_len / audio_len
        for label in self.encoder.labels:
            durations = df.loc[df["event_label"] == label, "duration"].dropna().to_numpy(dtype=float)
            durations = durations[durations > 0]
            if durations.size == 0:
                duration = None
                window = default_window
            else:
                if statistic == "median":
                    duration = float(np.median(durations))
                elif statistic == "mean":
                    duration = float(np.mean(durations))
                elif statistic.startswith("quantile:"):
                    q = float(statistic.split(":", 1)[1])
                    duration = float(np.quantile(durations, q))
                else:
                    duration = self._esp_duration(durations)
                window = round(duration * scaling_factor * seconds_to_frames)
                window = self._odd_window(window)
                window = min(max(window, min_window), max_window)
            windows.append(window)
            rows.append({
                "event_label": label,
                "event_count": int(durations.size),
                "duration_stat": duration if duration is not None else np.nan,
                "median_window": int(window),
            })

        self._median_filter = windows
        out_path = cfg.get("save_path")
        if out_path is None:
            out_path = os.path.join(self.config["generals"]["save_folder"], "classwise_median_windows.csv")
        pd.DataFrame(rows).to_csv(out_path, index=False)
        self.my_logger.logger.info(
            "classwise median filter enabled: min=%d, max=%d, mean=%.2f, saved=%s"
            % (min(windows), max(windows), float(np.mean(windows)), out_path)
        )
        return self._median_filter

    def train(self, epoch):
        self.net.train()
        n_train = len(self.train_loader)
        log_dict = OrderedDict([
            ("loss_total", 0),
            ("loss_class_strong", 0),
            ("loss_class_at_specific", 0),
            ("loss_hierarchy", 0),
            ("loss_hierarchy_feature", 0),
            ("loss_hierarchy_score", 0),
            ("hierarchy_feature_pairs", 0),
            ("hierarchy_score_pairs", 0),
            ("hierarchy_feature_weight", 0),
            ("hierarchy_score_weight", 0),
        ])

        tk = self._progress_bar(n_train)

        for batch in tk:
            wav, labels, _, _ = batch
            wav, labels = wav.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            feat, labels, labels_weak = self.preprocess(wav, labels)
            pred = self.net(feat, **self.config[self.net.get_model_name()]["train_kwargs"])
            pred = self.Pred(*pred)
            if torch.isnan(pred.strong).any():
                raise RuntimeError("Error: Out of range")

            if self.config[self.net.get_model_name()]["init_kwargs"]["at_param"]["out_type"] == "logit":
                if self.target_loss_indices is not None:
                    raise NotImplementedError("target_loss_labels_path currently supports sigmoid AT output only")
                at_out_logit = pred.other_dict["at_out"]
                loss_class_at_specific = self.ce_loss(
                    input=at_out_logit.transpose(1, 2),
                    target=multi_label_to_multi_class(labels_weak).transpose(1, 2),
                )
            elif self.config[self.net.get_model_name()]["init_kwargs"]["at_param"]["out_type"] == "sigmoid":
                target_indices = self._target_indices_for_device(labels.device)
                if target_indices is None:
                    loss_class_at_specific = self.supervised_loss(pred.other_dict["at_out"], labels_weak)
                else:
                    loss_class_at_specific = self.supervised_loss(
                        pred.other_dict["at_out"].index_select(1, target_indices),
                        labels_weak.index_select(1, target_indices),
                    )
            else:
                raise RuntimeError("Unknown output type for classification branch")

            target_indices = self._target_indices_for_device(labels.device)
            if target_indices is None:
                loss_class_strong = self.supervised_loss(pred.strong, labels)
            else:
                loss_class_strong = self.supervised_loss(
                    pred.strong.index_select(1, target_indices),
                    labels.index_select(1, target_indices),
                )
            loss_hierarchy_feature = pred.strong.new_zeros(())
            loss_hierarchy_score = pred.strong.new_zeros(())
            hierarchy_feature_pairs = 0
            hierarchy_score_pairs = 0
            hierarchy_feature_weight = self.hierarchy_feature_loss_weight
            hierarchy_score_weight = self.hierarchy_score_loss_weight
            if self.hierarchy_active_epochs is not None and epoch >= int(self.hierarchy_active_epochs):
                hierarchy_feature_weight = 0.0
                hierarchy_score_weight = 0.0
            if self.hierarchy_enabled:
                hierarchy_feat = pred.other_dict.get("hierarchy_feat", pred.other_dict.get("mask_feat"))
                if hierarchy_feat is not None and hierarchy_feature_weight > 0:
                    loss_hierarchy_feature, hierarchy_feature_pairs = self.hierarchy_helper.compute_feature_loss(
                        hierarchy_feat=hierarchy_feat,
                        weak_targets=labels_weak,
                    )
                if hierarchy_score_weight > 0:
                    loss_hierarchy_score, hierarchy_score_pairs = self.hierarchy_helper.compute_score_loss(
                        strong_scores=pred.strong,
                        weak_scores=self._at_scores_for_hierarchy(pred),
                        strong_targets=labels,
                        weak_targets=labels_weak,
                    )

            loss_hierarchy = (
                loss_hierarchy_feature * hierarchy_feature_weight
                + loss_hierarchy_score * hierarchy_score_weight
            )

            loss_total = (
                loss_class_strong
                + loss_class_at_specific * self.config["training"]["w_AT"]
                + loss_hierarchy
            )

            loss_total.backward()
            if self.config["training"]["clip_grad"]:
                torch.nn.utils.clip_grad_norm(self.net.parameters(), max_norm=20, norm_type=2)
            self.optimizer.step()
            self.scheduler.step()

            tk.set_postfix(
                loss=float(loss_total),
                strong=float(loss_class_strong),
                at=float(loss_class_at_specific),
                hier_feat=float(loss_hierarchy_feature),
                hier_score=float(loss_hierarchy_score),
                rel_lr=self.scheduler._get_scale(),
            )
            values = {
                "loss_total": loss_total,
                "loss_class_strong": loss_class_strong,
                "loss_class_at_specific": loss_class_at_specific,
                "loss_hierarchy": loss_hierarchy,
                "loss_hierarchy_feature": loss_hierarchy_feature,
                "loss_hierarchy_score": loss_hierarchy_score,
                "hierarchy_feature_pairs": hierarchy_feature_pairs,
                "hierarchy_score_pairs": hierarchy_score_pairs,
                "hierarchy_feature_weight": hierarchy_feature_weight,
                "hierarchy_score_weight": hierarchy_score_weight,
            }
            for key, value in values.items():
                if isinstance(value, torch.Tensor):
                    log_dict[key] += value.item() / n_train
                else:
                    log_dict[key] += value / n_train

        log_dict["lr_scaler"] = self.scheduler._get_scale()
        self.train_log(log_dict, epoch + 1)
        return

    def _progress_bar(self, n_train):
        from tqdm import tqdm

        return tqdm(self.train_loader, total=n_train, leave=False, desc="training processing")

    def _forward_eval(self, feat, pad_mask, kwargs):
        wrapped_model = getattr(self.net, "model", None)
        if self.eval_single_gpu and isinstance(wrapped_model, nn.DataParallel):
            return wrapped_model.module(input=feat, pad_mask=pad_mask, **kwargs)
        return self.net(input=feat, pad_mask=pad_mask, **kwargs)

    def mean_psds_per_type(self, psds_single):
        ret_dict = OrderedDict([
            ("common", []),
            ("rare", []),
        ])
        for event, psds in psds_single.items():
            event_type = self.type_dict.get(event)
            if event_type in ret_dict:
                ret_dict[event_type].append(psds)
        for key, values in ret_dict.items():
            ret_dict[key] = round(sum(values) / len(values), 4) if values else 0.0
        return ret_dict

    def validation(self, epoch):
        self.net.eval()
        map_kwargs = {
            "num_labels": len(self.eval_encoder.labels),
            "average": "macro",
        }
        metric_init = inspect.signature(torchmetrics.classification.MultilabelAveragePrecision.__init__)
        if "compute_on_step" in metric_init.parameters:
            map_kwargs["compute_on_step"] = False
        get_at_map = torchmetrics.classification.MultilabelAveragePrecision(**map_kwargs).to(self.device)
        val_scores_postprocessed_buffer = {}
        val_pred_dfs = []
        n_valid = len(self.val_loader)

        with torch.no_grad():
            from tqdm import tqdm

            tk = tqdm(self.val_loader, total=n_valid, leave=False, desc="validation processing")
            for batch in tk:
                wav, labels, pad_mask, idx, filename, file_path = batch
                wav, labels = wav.to(self.device), labels.to(self.device)
                feat = self.preprocess_eval(wav)
                pred = self._forward_eval(
                    feat=feat,
                    pad_mask=pad_mask,
                    kwargs=self.config[self.net.get_model_name()]["val_kwargs"],
                )
                pred = self.Pred(*pred)
                if self.config[self.net.get_model_name()]["init_kwargs"]["at_param"]["out_type"] == "logit":
                    weak_preds = multi_class_to_multi_label(F.softmax(pred.other_dict["at_out"], dim=-1))
                else:
                    weak_preds = pred.other_dict["at_out"]

                labels_weak = (torch.sum(labels, -1) >= 1).float()
                eval_weak_preds = self._select_eval_tensor(weak_preds)
                eval_labels_weak = self._select_eval_tensor(labels_weak)
                get_at_map(eval_weak_preds, eval_labels_weak.long().to(self.device))
                eval_strong = self._select_eval_tensor(pred.strong)
                _, scores_postprocessed_strong = batched_decode_preds(
                    strong_preds=eval_strong,
                    filenames=file_path,
                    encoder=self.eval_encoder,
                    weak_preds=None,
                    need_weak_mask=False,
                    filter=self._eval_median_filter(),
                )
                val_scores_postprocessed_buffer.update(scores_postprocessed_strong)
                if self.config["generals"].get("save_val_pred_tsv", True) or self.config["generals"].get("save_val_classwise_f1", True):
                    pred_dfs = decode_pred_batch_fast(
                        outputs=eval_strong,
                        weak_preds=torch.ones_like(eval_weak_preds),
                        filenames=file_path,
                        encoder=self.eval_encoder,
                        thresholds=[0.5],
                        median_filter=self._eval_median_filter(),
                    )
                    val_pred_dfs.append(pred_dfs[0.5])

        val_tsv, val_dur, val_audio_ids = self._filtered_eval_files(
            self.config["dataset"]["val_tsv"],
            self.config["dataset"]["val_dur"],
            "val",
        )
        val_scores_postprocessed_buffer = self._align_scores_to_audio_ids(
            val_scores_postprocessed_buffer,
            val_audio_ids,
            val_dur,
        )
        psds_sed_scores, psds_sed_scores_single = self.psds(
            val_scores_postprocessed_buffer,
            val_tsv,
            val_dur,
            save_dir=None,
            events_set=self.eval_events_set,
        )
        at_mAP = get_at_map.compute()
        obj_metric = psds_sed_scores
        partial_psds = self.mean_psds_per_type(psds_sed_scores_single)
        log_dict = OrderedDict([
            ("psds", psds_sed_scores),
            ("psds (common)", partial_psds["common"]),
            ("psds (rare)", partial_psds["rare"]),
            ("at mAP", at_mAP),
        ])
        psds_sed_scores_single = {k: round(v, 4) for k, v in psds_sed_scores_single.items()}
        with open(os.path.join(self.config["generals"]["save_folder"], "single_psds.json"), "w") as f:
            json.dump(OrderedDict(sorted(psds_sed_scores_single.items(), key=lambda x: x[1])), f, indent=4)

        if val_pred_dfs:
            pred_df = pd.concat(val_pred_dfs, ignore_index=True)
        else:
            pred_df = pd.DataFrame(columns=["event_label", "onset", "offset", "filename"])
        pred_df = self._filter_pred_df_to_audio_ids(pred_df, val_audio_ids)

        if self.config["generals"].get("save_val_pred_tsv", True):
            pred_tsv_path = os.path.join(self.config["generals"]["save_folder"], f"val_pred_events_epoch{epoch + 1}.tsv")
            pred_df.to_csv(pred_tsv_path, sep="\t", index=False)

        if self.config["generals"].get("save_val_classwise_f1", True):
            gt_df = pd.read_csv(val_tsv, sep="\t")
            event_metric = event_based_evaluation_df(gt_df, pred_df, t_collar=0.200, percentage_of_length=0.2)
            segment_metric = segment_based_evaluation_df(gt_df, pred_df, time_resolution=1.0)
            event_classwise = event_metric.results()["class_wise"]
            segment_classwise = segment_metric.results()["class_wise"]
            classwise_df = self._classwise_f1_dataframe(event_classwise, segment_classwise)
            classwise_path = os.path.join(self.config["generals"]["save_folder"], f"val_classwise_f1_epoch{epoch + 1}.csv")
            classwise_df.to_csv(classwise_path, index=False)

        self.val_log(log_dict, epoch + 1)
        return obj_metric

    def test(self):
        self.net.eval()
        n_test = len(self.test_loader)
        filter_type = self.config["training"]["filter_type"] if "filter_type" in self.config["training"] else "median"

        if self.config["generals"]["test_on_public_eval"]:
            test_tsv = self.config["dataset"]["pubeval_tsv"]
            test_dur = self.config["dataset"]["pubeval_dur"]
        else:
            test_tsv = self.config["dataset"]["test_tsv"]
            test_dur = self.config["dataset"]["test_dur"]
        test_tsv, test_dur, test_audio_ids = self._filtered_eval_files(test_tsv, test_dur, "test")
        test_df = pd.read_csv(test_tsv, sep="\t")
        test_events_set = set(test_df["event_label"])
        extra_events = set(self.eval_encoder.labels) - test_events_set
        scores_dir = os.path.join(self.config["generals"]["save_folder"], "sed_scores")
        os.makedirs(scores_dir, exist_ok=True)

        with torch.no_grad():
            from tqdm import tqdm

            tk = tqdm(self.test_loader, total=n_test, leave=False, desc="test processing")
            save_pred_tsv = self.config["generals"].get("save_pred_tsv", False)
            pred_tsv_path = os.path.join(self.config["generals"]["save_folder"], "pred_events.tsv")
            if save_pred_tsv and (not os.path.exists(pred_tsv_path)):
                with open(pred_tsv_path, "w", encoding="utf-8") as f:
                    f.write("filename\tonset\toffset\tevent_label\n")
            written_score_ids = set()
            for batch in tk:
                wav, labels, pad_mask, idx, filename, path = batch
                wav, labels = wav.to(self.device), labels.to(self.device)
                feat = self.preprocess_eval(wav)
                pred = self._forward_eval(
                    feat=feat,
                    pad_mask=pad_mask,
                    kwargs=self.config[self.net.get_model_name()]["test_kwargs"],
                )
                pred = self.Pred(*pred)
                eval_strong = self._select_eval_tensor(pred.strong)
                _, scores = batched_decode_preds(
                    strong_preds=eval_strong,
                    filenames=path,
                    encoder=self.eval_encoder,
                    filter=self._eval_median_filter(),
                    weak_preds=None,
                    need_weak_mask=False,
                    filter_type=filter_type,
                )

                if extra_events:
                    for audio_id, df in scores.items():
                        scores[audio_id] = df.drop(extra_events, axis=1)
                if test_audio_ids is not None:
                    scores = {audio_id: df for audio_id, df in scores.items() if audio_id in test_audio_ids}
                written_score_ids.update(scores)
                sed_scores_eval.io.write_sed_scores(scores, scores_dir)
                if save_pred_tsv:
                    pred_dfs = decode_pred_batch_fast(
                        outputs=eval_strong,
                        weak_preds=torch.ones(
                            eval_strong.shape[0],
                            eval_strong.shape[1],
                            device=eval_strong.device,
                            dtype=eval_strong.dtype,
                        ),
                        filenames=path,
                        encoder=self.eval_encoder,
                        thresholds=[0.5],
                        median_filter=self._eval_median_filter(),
                    )
                    pred_df = self._filter_pred_df_to_audio_ids(pred_dfs[0.5], test_audio_ids)
                    pred_df.to_csv(pred_tsv_path, sep="\t", index=False, header=False, mode="a")

            if test_audio_ids is not None:
                missing_score_ids = test_audio_ids - written_score_ids
                if missing_score_ids:
                    dur_df = pd.read_csv(test_dur, sep="\t")
                    duration_by_id = {
                        self._audio_id(row["filename"]): row["duration"]
                        for _, row in dur_df.iterrows()
                    }
                    zero_scores = {
                        audio_id: self._zero_score_dataframe(duration_by_id.get(audio_id, self.config["feature"]["audio_max_len"]))
                        for audio_id in missing_score_ids
                    }
                    sed_scores_eval.io.write_sed_scores(zero_scores, scores_dir)

        psds, psds_single = self.psds(
            scores_dir,
            test_tsv,
            test_dur,
            save_dir=self.config["training"]["psds_folder"],
            events_set=test_events_set,
        )
        partial_psds = self.mean_psds_per_type(psds_single)
        log_dict = OrderedDict([
            ("psds/s", psds),
            ("psds (common)", partial_psds["common"]),
            ("psds (rare)", partial_psds["rare"]),
        ])
        psds_single = {k: round(v, 4) for k, v in psds_single.items()}
        with open(os.path.join(self.config["generals"]["save_folder"], "single_psds.json"), "w") as f:
            json.dump(OrderedDict(sorted(psds_single.items(), key=lambda x: x[1])), f, indent=4)
        self.test_log(log_dict)
        return
