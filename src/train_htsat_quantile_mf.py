from pathlib import Path
import math
import os
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.append(str(ROOT))

from recipes.audioset_strong.detect_any_sound.htsat.train import DASM_HTSAT_Trainer


class EventSpecificMedianFilterHTSATTrainer(DASM_HTSAT_Trainer):
    """DASM HTSAT trainer with only event-specific median filtering changed."""

    @staticmethod
    def _odd_window(value):
        value = max(1, int(value))
        return value + 1 if value % 2 == 0 else value

    @staticmethod
    def _round_frames(value, method):
        if method == "ceil":
            return int(math.ceil(value))
        if method == "floor":
            return int(math.floor(value))
        return int(round(value))

    @property
    def median_fiter(self):
        if hasattr(self, "_median_filter"):
            return self._median_filter

        cfg = self.config["training"].get("classwise_median_filter", {})
        if not cfg.get("enabled", False):
            return super().median_fiter

        pred_len = int(self.config["feature"]["pred_len"])
        audio_len = float(self.config["feature"]["audio_max_len"])
        seconds_to_frames = pred_len / audio_len
        scaling_factor = float(cfg.get("scaling_factor", 1.0 / 3.0))
        statistic = cfg.get("statistic", "median")
        rounding = cfg.get("rounding", "ceil")
        min_window = self._odd_window(int(cfg.get("min_window", 1)))
        max_window = self._odd_window(int(cfg.get("max_window", 101)))
        default_window = self._odd_window(
            int(self.config["training"]["median_window"] / 156 * pred_len)
        )

        source_tsv = Path(cfg.get("source_tsv") or self.config["dataset"]["strong_tsv"])
        if not source_tsv.is_absolute():
            source_tsv = ROOT / source_tsv

        events = pd.read_csv(source_tsv, sep="\t")
        events = events[events["event_label"].isin(self.encoder.labels)].copy()
        events["duration"] = (events["offset"] - events["onset"]).clip(lower=0)

        windows = []
        rows = []
        eps = float(cfg.get("eps", 1e-6))
        knee_min_percent = float(cfg.get("knee_min_percent", 0.05))
        knee_max_percent = float(cfg.get("knee_max_percent", 0.95))
        knee_min_count = int(cfg.get("knee_min_count", 5))
        knee_score_threshold = float(cfg.get("knee_score_threshold", 0.0))
        knee_reference_statistic = cfg.get("knee_reference_statistic", "mean")
        for label in self.encoder.labels:
            durations = events.loc[events["event_label"] == label, "duration"].dropna()
            durations = durations[durations > 0].to_numpy(dtype=float)
            knee_index = np.nan
            knee_score = np.nan
            knee_duration = np.nan
            knee_used = False
            duration_method = "default_no_events"
            duration_type = "no_events"
            if durations.size == 0:
                duration = np.nan
                duration_q25 = np.nan
                duration_q50 = np.nan
                duration_q75 = np.nan
                duration_mean = np.nan
                adaptive_lambda = np.nan
                raw_window = np.nan
                window = default_window
            else:
                if statistic == "median":
                    duration = float(np.median(durations))
                    duration_q25 = float(np.quantile(durations, 0.25))
                    duration_q50 = duration
                    duration_q75 = float(np.quantile(durations, 0.75))
                    duration_mean = float(np.mean(durations))
                    adaptive_lambda = 0.0
                    duration_method = "median"
                    duration_type = "single_statistic"
                elif statistic == "mean":
                    duration = float(np.mean(durations))
                    duration_q25 = float(np.quantile(durations, 0.25))
                    duration_q50 = float(np.quantile(durations, 0.50))
                    duration_q75 = float(np.quantile(durations, 0.75))
                    duration_mean = duration
                    adaptive_lambda = 0.0
                    duration_method = "mean"
                    duration_type = "single_statistic"
                elif statistic.startswith("quantile:"):
                    q = float(statistic.split(":", 1)[1])
                    duration = float(np.quantile(durations, q))
                    duration_q25 = float(np.quantile(durations, 0.25))
                    duration_q50 = float(np.quantile(durations, 0.50))
                    duration_q75 = float(np.quantile(durations, 0.75))
                    duration_mean = float(np.mean(durations))
                    adaptive_lambda = 0.0
                    duration_method = f"quantile_{q:g}"
                    duration_type = "single_statistic"
                elif statistic in ("q25_q50_q75_trimean", "tukey_trimean", "trimean"):
                    duration_q25 = float(np.quantile(durations, 0.25))
                    duration_q50 = float(np.quantile(durations, 0.50))
                    duration_q75 = float(np.quantile(durations, 0.75))
                    duration_mean = float(np.mean(durations))
                    adaptive_lambda = np.nan
                    duration = (duration_q25 + 2.0 * duration_q50 + duration_q75) / 4.0
                    duration_method = "trimean"
                    duration_type = "single_statistic"
                elif statistic in (
                    "duration_knee",
                    "knee",
                    "elbow",
                    "duration_knee_or_trimean",
                    "knee_or_trimean",
                    "duration_type_aware",
                    "type_aware",
                ):
                    duration_q25 = float(np.quantile(durations, 0.25))
                    duration_q50 = float(np.quantile(durations, 0.50))
                    duration_q75 = float(np.quantile(durations, 0.75))
                    duration_mean = float(np.mean(durations))
                    adaptive_lambda = np.nan
                    fallback_duration = duration_q50
                    fallback_method = "median_low_score"
                    if statistic in ("duration_knee_or_trimean", "knee_or_trimean"):
                        fallback_duration = (duration_q25 + 2.0 * duration_q50 + duration_q75) / 4.0
                        fallback_method = "trimean_low_score"
                    if statistic in ("duration_type_aware", "type_aware"):
                        short_q75 = float(cfg.get("type_short_q75_seconds", 1.0))
                        long_q75 = float(cfg.get("type_long_q75_seconds", 4.0))
                        if duration_q75 < short_q75:
                            duration = duration_q25
                            duration_method = "type_short_q25"
                            duration_type = "short"
                            raw_window = duration * seconds_to_frames * scaling_factor
                            window = self._odd_window(self._round_frames(raw_window, rounding))
                            window = min(max(window, min_window), max_window)
                            windows.append(int(window))
                            rows.append({
                                "event_label": label,
                                "event_count": int(durations.size),
                                "duration_seconds": duration,
                                "duration_q25_seconds": duration_q25,
                                "duration_q50_seconds": duration_q50,
                                "duration_q75_seconds": duration_q75,
                                "duration_mean_seconds": duration_mean,
                                "adaptive_lambda": adaptive_lambda,
                                "knee_index": knee_index,
                                "knee_score": knee_score,
                                "knee_duration_seconds": knee_duration,
                                "knee_used": knee_used,
                                "duration_type": duration_type,
                                "duration_method": duration_method,
                                "raw_window_frames": raw_window,
                                "median_window_frames": int(window),
                                "window_seconds": int(window) / seconds_to_frames,
                            })
                            continue
                        if duration_q75 < long_q75:
                            duration = (duration_q25 + 2.0 * duration_q50 + duration_q75) / 4.0
                            duration_method = "type_medium_trimean"
                            duration_type = "medium"
                            raw_window = duration * seconds_to_frames * scaling_factor
                            window = self._odd_window(self._round_frames(raw_window, rounding))
                            window = min(max(window, min_window), max_window)
                            windows.append(int(window))
                            rows.append({
                                "event_label": label,
                                "event_count": int(durations.size),
                                "duration_seconds": duration,
                                "duration_q25_seconds": duration_q25,
                                "duration_q50_seconds": duration_q50,
                                "duration_q75_seconds": duration_q75,
                                "duration_mean_seconds": duration_mean,
                                "adaptive_lambda": adaptive_lambda,
                                "knee_index": knee_index,
                                "knee_score": knee_score,
                                "knee_duration_seconds": knee_duration,
                                "knee_used": knee_used,
                                "duration_type": duration_type,
                                "duration_method": duration_method,
                                "raw_window_frames": raw_window,
                                "median_window_frames": int(window),
                                "window_seconds": int(window) / seconds_to_frames,
                            })
                            continue
                        fallback_duration = (duration_q25 + 2.0 * duration_q50 + duration_q75) / 4.0
                        fallback_method = "type_long_trimean_low_score"
                        duration_type = "long"
                    if durations.size < knee_min_count or float(np.max(durations) - np.min(durations)) <= eps:
                        duration = fallback_duration
                        knee_duration = duration_q50
                        duration_method = fallback_method.replace("low_score", "low_count_or_flat")
                        if statistic not in ("duration_type_aware", "type_aware"):
                            duration_type = "single_statistic"
                    else:
                        sorted_duration = np.sort(durations)
                        # Paper-style ESP: x-axis is event duration and y-axis is
                        # normalized cumulative occupation probability (CDF).
                        x = (sorted_duration - sorted_duration[0]) / max(sorted_duration[-1] - sorted_duration[0], eps)
                        y = (np.arange(sorted_duration.size, dtype=float) + 1.0) / float(sorted_duration.size)
                        # CDF curves with long tails rise quickly then flatten. The
                        # max distance above the diagonal estimates the inflection point.
                        score = y - x
                        lo = max(0, int(np.floor(knee_min_percent * (sorted_duration.size - 1))))
                        hi = min(sorted_duration.size, int(np.ceil(knee_max_percent * (sorted_duration.size - 1))) + 1)
                        if hi <= lo:
                            lo, hi = 0, sorted_duration.size
                        local_idx = int(np.argmax(score[lo:hi]))
                        knee_index = lo + local_idx
                        knee_score = float(score[int(knee_index)])
                        knee_duration = float(sorted_duration[int(knee_index)])
                        if knee_score >= knee_score_threshold:
                            # Use the dominant side around the knee. The default keeps
                            # the original before-knee mean rule; trimean variants are
                            # robust to local extremes in the selected side.
                            if knee_reference_statistic in ("side_trimean", "adaptive_side_trimean"):
                                knee_ratio = (float(knee_index) + 1.0) / float(sorted_duration.size)
                                if knee_ratio < 0.5:
                                    knee_body = sorted_duration[int(knee_index) + 1 :]
                                    if knee_body.size == 0:
                                        knee_body = sorted_duration
                                    knee_method_suffix = "knee_after_trimean"
                                else:
                                    knee_body = sorted_duration[: int(knee_index) + 1]
                                    knee_method_suffix = "knee_before_trimean"
                                body_q25 = float(np.quantile(knee_body, 0.25))
                                body_q50 = float(np.quantile(knee_body, 0.50))
                                body_q75 = float(np.quantile(knee_body, 0.75))
                                duration = (body_q25 + 2.0 * body_q50 + body_q75) / 4.0
                            else:
                                knee_body = sorted_duration[: int(knee_index) + 1]
                            if knee_reference_statistic in ("trimean", "tukey_trimean", "q25_q50_q75_trimean"):
                                body_q25 = float(np.quantile(knee_body, 0.25))
                                body_q50 = float(np.quantile(knee_body, 0.50))
                                body_q75 = float(np.quantile(knee_body, 0.75))
                                duration = (body_q25 + 2.0 * body_q50 + body_q75) / 4.0
                                knee_method_suffix = "knee_trimean"
                            elif knee_reference_statistic not in ("side_trimean", "adaptive_side_trimean"):
                                duration = float(np.mean(knee_body))
                                knee_method_suffix = "knee_mean"
                            knee_used = True
                            duration_method = (
                                f"type_long_{knee_method_suffix}"
                                if statistic in ("duration_type_aware", "type_aware")
                                else knee_method_suffix
                            )
                            if statistic not in ("duration_type_aware", "type_aware"):
                                duration_type = "single_statistic"
                        else:
                            duration = fallback_duration
                            duration_method = fallback_method
                            if statistic not in ("duration_type_aware", "type_aware"):
                                duration_type = "single_statistic"
                elif statistic in ("q50_q75_adaptive", "adaptive_q50_q75"):
                    duration_q25 = float(np.quantile(durations, 0.25))
                    duration_q50 = float(np.quantile(durations, 0.50))
                    duration_q75 = float(np.quantile(durations, 0.75))
                    duration_mean = float(np.mean(durations))
                    adaptive_lambda = (duration_mean - duration_q50) / max(duration_q75 - duration_q50, eps)
                    adaptive_lambda = float(np.clip(adaptive_lambda, 0.0, 1.0))
                    duration = duration_q50 + adaptive_lambda * (duration_q75 - duration_q50)
                    duration_method = "q50_q75_adaptive"
                    duration_type = "single_statistic"
                else:
                    raise ValueError(f"Unsupported classwise median filter statistic: {statistic}")
                raw_window = duration * seconds_to_frames * scaling_factor
                window = self._odd_window(self._round_frames(raw_window, rounding))
                window = min(max(window, min_window), max_window)

            windows.append(int(window))
            rows.append({
                "event_label": label,
                "event_count": int(durations.size),
                "duration_seconds": duration,
                "duration_q25_seconds": duration_q25,
                "duration_q50_seconds": duration_q50,
                "duration_q75_seconds": duration_q75,
                "duration_mean_seconds": duration_mean,
                "adaptive_lambda": adaptive_lambda,
                "knee_index": knee_index,
                "knee_score": knee_score,
                "knee_duration_seconds": knee_duration,
                "knee_used": knee_used,
                "duration_type": duration_type,
                "duration_method": duration_method,
                "raw_window_frames": raw_window,
                "median_window_frames": int(window),
                "window_seconds": int(window) / seconds_to_frames,
            })

        self._median_filter = windows
        out_path = cfg.get("save_path")
        if out_path is None:
            out_path = os.path.join(
                self.config["generals"]["save_folder"],
                "classwise_median_windows.csv",
            )
        pd.DataFrame(rows).to_csv(out_path, index=False)
        self.my_logger.logger.info(
            "event-specific median filter enabled: statistic=%s, scaling=%.6f, "
            "rounding=%s, min=%d, max=%d, mean=%.2f, saved=%s"
            % (
                statistic,
                scaling_factor,
                rounding,
                min(windows),
                max(windows),
                float(np.mean(windows)),
                out_path,
            )
        )
        return self._median_filter
