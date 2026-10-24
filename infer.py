"""Inference-only OCL+CAW+CAMF sound-event detection."""

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
from scipy.ndimage import median_filter

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.models.detect_any_sound.detect_any_sound_htast import DASM_HTSAT


def read_labels(path):
    with path.open(encoding="utf-8") as handle:
        mapping = json.load(handle)
    labels = [None] * len(mapping)
    for label, index in mapping.items():
        if not isinstance(index, int) or index < 0 or index >= len(labels) or labels[index] is not None:
            raise ValueError("Invalid or duplicate class index in label dictionary")
        labels[index] = label
    if any(label is None for label in labels):
        raise ValueError("Missing class index in label dictionary")
    return labels


def read_windows(path, labels):
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    by_label = {row["event_label"]: int(row["median_window_frames"]) for row in rows}
    if len(by_label) != len(rows) or set(by_label) != set(labels):
        raise ValueError("CAMF window labels do not match the model classes")
    windows = [by_label[label] for label in labels]
    if any(window < 1 or window % 2 != 1 for window in windows):
        raise ValueError("CAMF windows must be positive odd integers")
    return windows


def iter_files(path):
    if path.is_file():
        if path.suffix.lower() != ".wav":
            raise ValueError("Input must be a WAV file or a directory of WAV files")
        return [(path, path.name)]
    if not path.is_dir():
        raise FileNotFoundError(path)
    files = sorted(p for p in path.rglob("*") if p.is_file() and p.suffix.lower() == ".wav")
    if not files:
        raise ValueError("No WAV files found under " + str(path))
    return [(p, p.relative_to(path).as_posix()) for p in files]


def chunk_starts(total_samples, clip_samples, hop_samples):
    if total_samples <= clip_samples:
        return [0]
    starts = list(range(0, total_samples - clip_samples + 1, hop_samples))
    last = total_samples - clip_samples
    if starts[-1] != last:
        starts.append(last)
    return starts


def decode_chunk(scores, windows, class_indices, threshold, frame_seconds, start_seconds,
                 valid_seconds, valid_frames):
    events = []
    for class_index in class_indices:
        smoothed = median_filter(scores[:, class_index], size=windows[class_index])
        active = smoothed[:valid_frames] > threshold
        edges = np.diff(np.pad(active.astype(np.int8), (1, 1)))
        for onset_frame, offset_frame in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
            onset = start_seconds + onset_frame * frame_seconds
            offset = start_seconds + min(offset_frame * frame_seconds, valid_seconds)
            if onset < offset:
                events.append((class_index, onset, offset))
    return events


def merge_events(events, frame_seconds):
    merged = []
    for class_index, onset, offset in sorted(events, key=lambda row: (row[0], row[1], row[2])):
        if merged and merged[-1][0] == class_index and onset <= merged[-1][2] + frame_seconds / 2:
            merged[-1] = (class_index, merged[-1][1], max(offset, merged[-1][2]))
        else:
            merged.append((class_index, onset, offset))
    return sorted(merged, key=lambda row: (row[1], row[0], row[2]))


def infer_file(path, model, device, audio_cfg, windows, class_indices, threshold, hop_seconds):
    target_sr = audio_cfg["sample_rate"]
    clip_samples = int(audio_cfg["clip_seconds"] * target_sr)
    hop_samples = int(hop_seconds * target_sr)
    pred_len = audio_cfg["pred_len"]
    frame_seconds = audio_cfg["hop_samples"] * audio_cfg["net_subsample"] / target_sr
    events = []
    with sf.SoundFile(str(path)) as audio:
        native_sr = audio.samplerate
        total_samples = math.ceil(len(audio) * target_sr / native_sr)
        if total_samples == 0:
            return []
        for start in chunk_starts(total_samples, clip_samples, hop_samples):
            start_native = round(start * native_sr / target_sr)
            audio.seek(min(start_native, len(audio)))
            read_native = math.ceil(clip_samples * native_sr / target_sr) + 2
            wav = audio.read(read_native, dtype="float32", always_2d=True).mean(axis=1)
            if native_sr != target_sr:
                wav = librosa.resample(wav, orig_sr=native_sr, target_sr=target_sr)
            wav = wav[:clip_samples]
            valid = len(wav)
            if valid == 0:
                continue
            padded = np.pad(wav, (0, clip_samples - valid))
            valid_frames = min(pred_len, math.ceil(valid / (audio_cfg["hop_samples"] * audio_cfg["net_subsample"])))
            pad_mask = torch.arange(pred_len, device=device).unsqueeze(0) >= valid_frames
            with torch.inference_mode():
                features = model.get_feature_extractor()(torch.from_numpy(padded).unsqueeze(0).to(device))
                strong, _, _ = model(input=features, pad_mask=pad_mask, encoder_win=False, temp_w=0.5)
            scores = strong[0].float().cpu().numpy().T
            events.extend(decode_chunk(scores, windows, class_indices, threshold, frame_seconds,
                                       start / target_sr, valid / target_sr, valid_frames))
    return merge_events(events, frame_seconds)


def main():
    parser = argparse.ArgumentParser(description="Run OCL+CAW+CAMF inference on WAV audio")
    parser.add_argument("--input", type=Path, required=True, help="WAV file or directory")
    parser.add_argument("--output", type=Path, required=True, help="Output events TSV")
    parser.add_argument("--device", default="auto", help="auto, cpu, or cuda:0")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--hop_seconds", type=float, default=5.0, help="Hop for audio longer than 10 s")
    parser.add_argument("--classes", nargs="*", help="Exact class labels; default: all 407")
    args = parser.parse_args()
    if not 0 <= args.threshold <= 1:
        parser.error("--threshold must be between 0 and 1")
    if not 0 < args.hop_seconds <= 10:
        parser.error("--hop_seconds must be in (0, 10]")
    if args.device == "auto":
        args.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable")

    with (ROOT / "configs/inference.json").open(encoding="utf-8") as handle:
        cfg = json.load(handle)
    labels = read_labels(ROOT / "assets/metadata/labeldict_ontology_audioset_strong.json")
    windows = read_windows(ROOT / "assets/windows/classwise_median_windows.csv", labels)
    if cfg["model"]["init_kwargs"]["class_num"] != len(labels):
        raise ValueError("Class count in config does not match label dictionary")
    if args.classes:
        unknown = set(args.classes) - set(labels)
        if unknown:
            parser.error("Unknown class labels: " + ", ".join(sorted(unknown)))
        class_indices = [i for i, label in enumerate(labels) if label in args.classes]
    else:
        class_indices = range(len(labels))

    cfg["model"]["init_kwargs"]["at_param"]["query"] = str(ROOT / "assets/queries/dasm_query_407x768.pt")
    model = DASM_HTSAT(**cfg["model"]["init_kwargs"])
    state = torch.load(ROOT / "weights/ocl_caw_best_model.pt", map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    files = iter_files(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    total_events = 0
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(("filename", "onset", "offset", "event_label"))
        for path, filename in files:
            events = infer_file(path, model, device, cfg["audio"], windows, class_indices,
                                args.threshold, args.hop_seconds)
            for class_index, onset, offset in events:
                writer.writerow((filename, f"{onset:.3f}", f"{offset:.3f}", labels[class_index]))
            total_events += len(events)
            print(f"{filename}: {len(events)} events", flush=True)
    print(f"Saved {total_events} events from {len(files)} files to {args.output}")


if __name__ == "__main__":
    main()
