# ontology-guided-sed

This folder contains the frozen OCL+CAW DASM checkpoint and CAMF postprocessing
needed to detect 407 AudioSet-Strong event classes. It does not include training
data, training code, validation labels, or PSDS evaluation.

## Interactive demo

Explore the seven precomputed sound event examples in the [browser demo](https://hxx-source.github.io/ontology-guided-sed/). The static page lives in docs/ and requires no local Python setup.

## Setup

Tested with Python 3.8, PyTorch 2.0.1+cu118, and the versions in
`requirements.txt`. For a new CUDA 11.8 environment, install the matching
PyTorch build first, then install the remaining dependencies:

```bash
python -m pip install torch==2.0.1+cu118 torchaudio==2.0.2+cu118 \
  torchvision==0.15.2+cu118 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements.txt
```

## Inference

```bash
cd ontology-guided-sed
CUDA_VISIBLE_DEVICES=0 python infer.py \
  --input /path/to/audio.wav --output results/events.tsv --device cuda:0
```

For all WAV files in a directory, pass the directory to `--input`. Long files
are processed in overlapping 10-second windows (5-second hop by default).
`--classes Cough Snoring` restricts the written detections; the model still
computes all 407 class scores. CPU inference is available with `--device cpu`.

The output is a tab-separated file with `filename`, `onset`, `offset`, and
`event_label` columns. Times are in seconds. The default threshold is 0.5.
The class-specific median windows in `assets/windows/` are fixed from training
data. Overlapping detections from adjacent windows are merged per class.

This is **inference**, not a PSDS benchmark. Scores from sliding windows on
long recordings are not directly equivalent to the original 10-second
AudioSet-Strong validation protocol.

## Repository contents

- `infer.py`: WAV loading, model inference, CAMF filtering, event decoding.
- `src/models/`: model architecture and only its import dependencies.
- `weights/ocl_caw_best_model.pt`: frozen checkpoint.
- `assets/queries/`: 407 audio query embeddings.
- `assets/metadata/` and `assets/windows/`: label order and CAMF windows.
- `configs/inference.json`: inference model settings.

The 160 MB checkpoint must be uploaded with Git LFS when publishing on GitHub;
`.gitattributes` already marks it for LFS. Install Git LFS before `git add`.
Only load the bundled `.pt` files from a trusted source; PyTorch 2.0 uses
pickle deserialization for checkpoints.

Source model code and release materials are covered by `LICENSE`.
