# ontology-guided-sed

Reference implementation for **Ontology-Guided Knowledge Transfer and Class-Adaptive Post-Processing for Open-Vocabulary Sound Event Detection**.

This repository contains the OCL+CAW training setup and CAMF post-processing setup used in our AudioSet-Strong experiments. It releases the code, metadata, class-wise analysis files, query tensors, and model checkpoints needed to reproduce the reported results, subject to the original licenses of AudioSet and third-party pretrained models.

## Highlights

- **OCL**: ontology consistency learning over selected AudioSet parent-child relations.
- **CAW**: class-adaptive pair weighting using child-parent duration ratio, parent support, and query cosine similarity.
- **CAMF**: class-adaptive median filtering using training-set event duration statistics and knee/side-trimean window selection.
- **Released checkpoints**: DASM audio-query baseline, HTS-AT/MGA initialization, and final OCL+CAW model checkpoints are provided through Git LFS.

## Main Results

AudioSet-Strong validation, PSDS1 setting: `dtc=0.7`, `gtc=0.7`, `cttc=None`, `alpha_ct=0`, `alpha_st=0`, `max_efpr=100`.

| System | PSDS | Non-rare | Rare |
| --- | ---: | ---: | ---: |
| A0 DASM baseline | 49.73 | 53.21 | 37.50 |
| A1 + OCL | 50.01 | 53.26 | 38.49 |
| A2 + OCL + CAW | 50.22 | 53.33 | 39.24 |
| A3 + OCL + CAW + CAMF | 51.26 | 54.27 | 40.68 |

The final CAMF class-wise window table is included at `assets/windows/classwise_median_windows.csv`.

## Repository Layout

```text
.
├── assets/
│   ├── metadata/      # ontology, label dictionary, class statistics, AudioSet-Strong TSV metadata
│   ├── pairs/         # final parent-child pairs and CAW components
│   ├── queries/       # 407-class DASM audio-query tensor
│   ├── results/       # final PSDS summaries and class-wise analysis files
│   └── windows/       # final CAMF class-wise median windows
├── checkpoints/       # released model checkpoints, tracked with Git LFS
├── configs/           # training and evaluation configuration files
├── docs/              # supplementary documentation
├── recipes/           # DASM training/evaluation recipes
├── scripts/           # runnable training and evaluation entry scripts
└── src/               # DASM source plus OCL/CAW/CAMF extensions
```

## Installation

```bash
conda create -n ontology-sed python=3.10 -y
conda activate ontology-sed
pip install -r requirements.txt
```

For CUDA builds, install a PyTorch version matching your driver before installing the remaining packages.

## Data Preparation

This repository includes metadata TSVs, ontology files, pair selections, query tensors, results, and checkpoints. It does **not** include the original AudioSet audio files.

Prepare the audio folders as:

```text
data/audioset_strong/train/
data/audioset_strong/val/
```

If your audio is stored elsewhere, update `dataset.strong_folder` and `dataset.val_folder` in `configs/train_ocl_caw.yaml` and `configs/eval_camf.yaml`.

## Checkpoints

Large checkpoint files are stored under `checkpoints/` and tracked with Git LFS. After cloning, make sure Git LFS is installed and pull the checkpoint files:

```bash
git lfs install
git lfs pull
```

Included checkpoints:

- `checkpoints/htast_mga.pt`: HTS-AT/MGA initialization checkpoint.
- `checkpoints/dasm_audio_query_baseline.pt`: DASM audio-query baseline checkpoint used to initialize OCL+CAW training.
- `checkpoints/ocl_caw_best_model.pt`: final OCL+CAW checkpoint used with CAMF.

## Train OCL+CAW

```bash
bash scripts/train_ocl_caw.sh 0
```

The script runs:

```bash
python src/main_htsat_hier.py   --config_dir configs/train_ocl_caw.yaml   --save_folder runs/ocl_caw   --gpu 0
```

The final OCL+CAW loss uses:

- `loss_weight: 0.1`
- `target_weight_mode: adaptive_pair`
- `require_positive_ancestor: true`
- `60` selected child-parent pairs covering `59` child classes
- adaptive pair weight = duration weight x parent-support weight x cosine-band weight

## Evaluate With CAMF

```bash
bash scripts/eval_camf.sh 0
```

The script runs:

```bash
python src/run_val_pred_psds_quantile_mf.py   --config configs/eval_camf.yaml   --save_folder runs/camf   --model_path checkpoints/ocl_caw_best_model.pt   --gpu 0
```

CAMF uses:

- `statistic: duration_knee_or_trimean`
- `knee_reference_statistic: side_trimean`
- `knee_score_threshold: 0.2`
- `knee_min_count: 20`
- `rounding: ceil`
- `min_window: 1`, `max_window: 101`
- `scaling_factor: 1/3`

## Citation

If this repository helps your work, please cite the associated paper and this code release. A `CITATION.cff` file is provided for GitHub citation metadata.

## License

Code is released under the MIT License. AudioSet audio, metadata, and third-party model weights remain subject to their original licenses and terms.
