from typing import Optional, Tuple

import numpy as np
import pandas as pd
import librosa
import soundfile as sf
import torch
import torch.nn as nn
import torchaudio
from torch import Tensor

from src.codec.encoder import Encoder
from src.models.transformer.conformer import ConvolutionModule
import src.preprocess.dataset as dataset_module
import src.preprocess.feats_extraction as feats_module


_AUDIO_LOAD_WARNED = set()


def _patched_conformer_conv_forward(
    self,
    x: Tensor,
    cache: Optional[Tensor] = None,
    right_context: int = 0,
    src_key_padding_mask: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor]:
    x = x.permute(1, 2, 0).contiguous()
    x = self.pointwise_conv1(x.contiguous())
    x = nn.functional.glu(x, dim=1)

    if src_key_padding_mask is not None:
        x.masked_fill_(src_key_padding_mask.unsqueeze(1).expand_as(x), 0.0)
    if self.causal and self.lorder > 0:
        if cache is None:
            x = nn.functional.pad(x, (self.lorder, 0), "constant", 0.0)
        else:
            assert not self.training, "Cache should be None in training time"
            assert cache.size(0) == self.lorder
            x = torch.cat([cache.permute(1, 2, 0).contiguous(), x], dim=2)
            if right_context > 0:
                cache = x.permute(2, 0, 1).contiguous()[
                    -(self.lorder + right_context):(-right_context),
                    ...,
                ]
            else:
                cache = x.permute(2, 0, 1).contiguous()[-self.lorder:, ...]

    x = self.depthwise_conv(x.contiguous())
    x = x.permute(0, 2, 1).contiguous()
    x = self.norm(x)
    x = x.permute(0, 2, 1).contiguous()
    x = self.activation(x)
    x = self.pointwise_conv2(x.contiguous())

    if cache is None:
        cache = torch.empty(0, device=x.device)

    return x.permute(2, 0, 1).contiguous(), cache


def apply_runtime_patches():
    ConvolutionModule.forward = _patched_conformer_conv_forward
    Encoder.encode_strong_df = _patched_encode_strong_df
    Encoder.encode_weak = _patched_encode_weak
    feats_module.waveform_modification = _patched_waveform_modification
    dataset_module.waveform_modification = _patched_waveform_modification


def configure_runtime(disable_cudnn=True):
    if disable_cudnn:
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def _patched_encode_strong_df(self, events_df):
    true_labels = np.zeros((self.n_frames, len(self.labels)))
    label_to_idx = {label: idx for idx, label in enumerate(self.labels)}
    for _, row in events_df.iterrows():
        event_label = row["event_label"]
        if pd.isna(event_label):
            continue
        label_idx = label_to_idx.get(event_label)
        if label_idx is None:
            continue
        onset = round(self._time_to_frame(row["onset"]))
        offset = round(np.ceil(self._time_to_frame(row["offset"])))
        true_labels[onset:offset, label_idx] = 1
    return true_labels


def _patched_encode_weak(self, events):
    labels = np.zeros((len(self.labels)))
    if len(events) == 0:
        return labels
    label_to_idx = {label: idx for idx, label in enumerate(self.labels)}
    for event in events:
        label_idx = label_to_idx.get(event)
        if label_idx is not None:
            labels[label_idx] = 1
    return labels


def _patched_waveform_modification(filepath, pad_to, encoder):
    duration = pad_to / encoder.sr
    wav = None
    native_sr = encoder.sr
    try:
        with sf.SoundFile(filepath) as audio_file:
            native_sr = audio_file.samplerate
            frames = min(len(audio_file), int(round(duration * native_sr)))
            wav = audio_file.read(frames=frames, dtype="float32", always_2d=False)
    except Exception:
        wav = None

    if wav is None:
        try:
            info = torchaudio.info(filepath)
            frames = int(round(duration * info.sample_rate))
            wav_tensor, native_sr = torchaudio.load(filepath, frame_offset=0, num_frames=frames)
            if wav_tensor.ndim > 1:
                wav_tensor = wav_tensor.mean(dim=0)
            else:
                wav_tensor = wav_tensor.squeeze(0)
            if native_sr != encoder.sr:
                wav_tensor = torchaudio.functional.resample(wav_tensor, native_sr, encoder.sr)
            wav = wav_tensor.cpu().numpy()
        except Exception:
            wav = None

    if wav is None:
        try:
            wav, _ = librosa.load(filepath, sr=encoder.sr, duration=duration)
        except Exception:
            wav = np.zeros(pad_to, dtype=np.float32)
            if filepath not in _AUDIO_LOAD_WARNED:
                _AUDIO_LOAD_WARNED.add(filepath)
                print(f"[audio-warning] failed to read {filepath}, replaced with silence")

    wav = np.asarray(wav, dtype=np.float32)
    if wav.ndim > 1:
        wav = np.mean(wav, axis=-1)
    if native_sr != encoder.sr and len(wav) > 0:
        wav = librosa.resample(wav, orig_sr=native_sr, target_sr=encoder.sr)
    if len(wav) < pad_to:
        pad_from = len(wav)
        wav = np.pad(wav, (0, pad_to - len(wav)), mode="constant")
    else:
        wav = wav[:pad_to]
        pad_from = pad_to
    pad_idx = np.ceil(encoder._time_to_frame(pad_from / encoder.sr))
    pad_mask = torch.arange(encoder.n_frames) >= pad_idx
    wav = torch.from_numpy(wav).float()
    return wav, pad_mask
