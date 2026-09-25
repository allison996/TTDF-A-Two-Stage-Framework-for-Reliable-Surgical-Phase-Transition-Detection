# TTDF: A Two-Stage Framework for Reliable Surgical Phase Transition Detection

**MICCAI 2026 workshop paper** · Yushi Guo, Pietro Valdastri, Duygu Sarikaya

University of Leeds · STORM Lab UK

TTDF turns predictions from a frozen online phase recognizer into reliable surgical workflow transition events. **Transition Candidate Extraction (TCE)** removes short phase jitter and workflow-illegal switches. **Transition Candidate Verification (TCV)** then evaluates the remaining candidates using phase-posterior shifts and frozen visual-change cues. The paper evaluates emitted events with ordered phase-pair-aware, one-to-one matching.

## Prerequisites

- Python 3.10+; the code was checked locally with Python 3.12.4, NumPy 2.2.4, and PyTorch 2.12.0.
- `requirements.txt` lists only NumPy and PyTorch because they are the only third-party packages imported by the released TTDF code. The separate frozen DINOv2/MS-TCN recognizer and its dependencies are not part of this repository.
- Install dependencies from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Training data

- Obtain [Cholec80 from the CAMMA dataset page](https://camma.unistra.fr/datasets/) and follow its dataset terms. The paper uses **32 training, 8 validation, and 40 test videos**, sampled at 1 frame per second.
- TTDF takes the outputs of a **frozen causal phase recognizer** as input. In the paper this is DINOv2 CLS features followed by a causal MS-TCN. Recognizer training and feature extraction are separate from the TTDF code released here.
- A *prediction trajectory* means the recognizer's predictions over time for **one video**; it does not mean instrument motion. The Python option `--trajectory-dir` uses this term for the saved recognizer outputs. Supply one matching `.npz` file per video in each `train`, `val`, and `test` directory:

| File | Required arrays |
| --- | --- |
| `recognizer_outputs/train/video01.npz` (and `val/`, `test/`) | `labels [T]`: ground-truth phase IDs; `predictions [T]`: recognizer phase IDs; `probabilities [T, 7]`: phase probabilities |
| `features/train/video01.npz` (and `val/`, `test/`) | `features [T, D]`: frozen visual features; `labels [T]`: ground-truth phase IDs |

Here `T` is the number of sampled frames in that video. For the paper's DINOv2 CLS features, `D = 384`. The frame rate, frame order, and filenames must match between the two directories. Workflow-legal phase pairs are estimated **only from training annotations**. Keep Cholec80 data and generated arrays outside Git.

**Which files must be shared?** TTDF training reads the `.npz` arrays above, not the phase recognizer's `.pt` weights. A `.pt` file is a PyTorch model checkpoint: this repository creates `tcv_model.pt` when TCV training finishes. Readers need either aligned recognizer outputs and features, or the recognizer code, weights, and export procedure to generate them. These experimental inputs are not bundled here; a TCV checkpoint by itself would not replace them.

## Train and obtain test results

This is the main command. It trains TCV with dwell duration 5, an 8-frame pre-candidate context, a causal 13-frame window, a one-layer TCN, and 50 epochs. It selects the retain threshold on validation data **and evaluates the test set automatically**.

```bash
bash scripts/train.sh /path/to/recognizer_outputs /path/to/features outputs/ttdf cuda
```

Read the test metrics in `outputs/ttdf/summary.json`; the output directory also contains `tcv_model.pt` and trigger tables. There is no need to run another script after training to obtain test metrics. To run on CPU, replace `cuda` with `cpu`. This command runs one experiment. The paper table averages five independent runs; to recompute that aggregate, repeat the command with a different `TTDF_SEED` value and output directory each time.

## Optional: evaluate a saved model again

Use `evaluate.sh` only when you want to evaluate an already trained model again, without retraining. It takes the `tcv_model.pt` checkpoint **produced by the main training command**. No pretrained checkpoint is bundled with this repository.

```bash
bash scripts/evaluate.sh \
  outputs/ttdf/tcv_model.pt \
  /path/to/recognizer_outputs/test \
  /path/to/features \
  outputs/ttdf/recheck cpu
```

This writes `metrics.json` and `trigger_table.csv`. For the raw phase-change baseline:

```bash
python event_commit_metrics.py \
  --trajectory-dir /path/to/recognizer_outputs/test \
  --policy raw --tolerance 30 \
  --output outputs/ttdf/raw_events.json
```

The data and frozen recognizer outputs used for the paper are not part of this repository, so the reported numbers require those inputs. See the publication for the full experimental protocol.

## Main files

| File | Purpose |
| --- | --- |
| `scripts/train.sh` | Main command: train TCV, select a validation threshold, and evaluate the test set. |
| `scripts/evaluate.sh` | Optional: re-evaluate a saved `tcv_model.pt` without training again. |
| `train_transition_reliability.py` | Python implementation called by `train.sh`. |
| `evaluate_checkpoint.py` | Python implementation called by `evaluate.sh`. |
| `tests/test_core.py` | Small unit tests for transition filtering and cue calculations; not part of training. |

## Results

**Table 1. Progressive event-level transition detection on Cholec80.** Values below are from the paper; the test script computes the metrics from the supplied trajectories.

| Method | Precision ↑ | Recall ↑ | F1 ↑ | False/GT ↓ | Dup/GT ↓ | Delay50 ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Raw changes | 0.093 | 0.672 | 0.163 | 6.570 | 0.332 | 0.0 s |
| + Dwell | 0.209 | 0.643 | 0.316 | 2.430 | 0.072 | 5.0 s |
| + Legality (TCE) | 0.321 | 0.643 | 0.428 | 1.357 | 0.072 | 5.0 s |
| + TCV (TTDF) | 0.470 | 0.531 | **0.497** | **0.603** | **0.013** | 6.3 s |

**Table 2. TCV cue ablation.** Five-seed mean ± standard deviation where reported in the paper.

| Cues | Emitted events ↓ | F1 ↑ | False/GT ↓ | Dup/GT ↓ |
| --- | ---: | ---: | ---: | ---: |
| Posterior L1 | 245.6 ± 61.0 | 0.475 ± 0.025 | 0.557 ± 0.173 | 0.013 |
| Posterior JS | 287.2 ± 76.4 | 0.484 ± 0.017 | 0.684 ± 0.245 | 0.020 ± 0.015 |
| Posterior L1 + JS | 244.2 ± 18.0 | 0.479 ± 0.012 | 0.551 ± 0.047 | 0.013 |
| Visual | 468.0 ± 4.5 | 0.426 ± 0.005 | 1.354 ± 0.008 | 0.071 ± 0.002 |
| Posterior + visual | 266.6 ± 30.9 | **0.497 ± 0.006** | 0.603 ± 0.094 | 0.013 |

## Citation

```bibtex
@inproceedings{guo2026ttdf,
  title  = {TTDF: A Two-Stage Framework for Reliable Surgical Phase Transition Detection},
  author = {Guo, Yushi and Valdastri, Pietro and Sarikaya, Duygu},
  year   = {2026},
  note   = {MICCAI workshop paper}
}
```

The workshop proceedings metadata will replace the provisional venue note when available. Contact: scyg@leeds.ac.uk. Code is licensed under [MIT](LICENSE); the Cholec80 dataset has its own terms.
