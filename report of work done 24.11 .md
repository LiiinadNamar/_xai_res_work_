## Goal

1. **Fix ReplayBG.** The old twin was fitted once, on day 1, and replayed on every later day.
On a day it was not fitted on, ReplayBG runs open-loop and glucose drifts to 0 or 10^16.
Now each day gets its own MAP fit. State carries over only into the next calendar day.
Meals are labelled B, L or D by clock time. A day is dropped if the replay leaves 20–500 mg/dL or its MARD is above 30%.
2. **Add a naive baseline.** Score = 1 if glucose_min of the past hour < 70 mg/dL, otherwise 0.
No training. This shows how much a model adds over "already low means hypo next hour".
3. **Add a time split.** Within each patient, the first 80% of windows train and the last 20% test.
GroupKFold tests on unseen patients. The time split tests on the future of known patients, which is how a personal alert would be used.

## Datasets (this run)

| Dataset | Source | Windows | Patients | Window hypo rate |
|---|---|---:|---:|---:|
| simglucose | simulator | 28,868 | 28 | 31.6% |
| HUPA-UCM | real CGM | 25,746 | 25 | 11.6% |
| ReplayBG | per-day twin on HUPA days | 6,579 | 21 | 8.5% |

ReplayBG is smaller than before (17,811 → 6,579 windows) because the rejected days are gone. Most of those windows came from exploded replays.

## Protocols

- **GroupKFold(5)** on patient. Every window of a patient is in train or in test. Mean ± std over 5 folds.
- **Time split.** Per patient, windows are sorted by *time*, then 80% train / 20% test. A patient with fewer than 24 test windows is dropped (one ReplayBG patient). It is a single split, so there is no std.
- Scaling is fitted on train only. Both protocols use the same models.

| Dataset | Train | Test | Test patients | Test hypo |
|---|---:|---:|---:|---:|
| simglucose | 23,100 | 5,768 | 28 | 33.0% |
| HUPA-UCM | 20,595 | 5,151 | 25 | 12.4% |
| ReplayBG | 5,189 | 1,295 | 20 | 8.8% |

## Results — AUPRC

| Dataset | Model | GroupKFold | Time |
|---|---|---:|---:|
| simglucose | LR | 0.917 ± 0.027 | 0.930 |
| simglucose | SVC | 0.917 ± 0.019 | 0.931 |
| simglucose | GRU | 0.938 ± 0.018 | 0.948 |
| simglucose | Transformer | 0.932 ± 0.018 | 0.943 |
| simglucose | naive | 0.644 ± 0.067 | 0.659 |
| HUPA-UCM | LR | 0.695 ± 0.132 | 0.711 |
| HUPA-UCM | SVC | 0.743 ± 0.130 | 0.808 |
| HUPA-UCM | GRU | 0.830 ± 0.097 | 0.854 |
| HUPA-UCM | Transformer | 0.848 ± 0.089 | 0.880 |
| HUPA-UCM | naive | 0.401 ± 0.100 | 0.455 |
| ReplayBG | LR | 0.776 ± 0.060 | 0.786 |
| ReplayBG | SVC | 0.805 ± 0.045 | 0.811 |
| ReplayBG | GRU | 0.827 ± 0.045 | 0.809 |
| ReplayBG | Transformer | 0.837 ± 0.036 | 0.854 |
| ReplayBG | naive | 0.456 ± 0.090 | 0.442 |

### Reading the results

- **ReplayBG is fixed.** GroupKFold went from 0.36 ± 0.31 (old twin) to 0.78–0.84 ± 0.04–0.06.
Every fold is now in 0.67–0.87, where before one fold was 0.91 and others were 0.12–0.31.
The ranking matches HUPA (sequence models above tabular), so the twin now behaves like the real data it was built from.
- **The naive baseline is far below every model.** It is 0.40–0.46 on the real and twin data and 0.64 on simglucose.
Even LR adds about +0.3 on HUPA. On simglucose, naive alone already gets 0.64, so that dataset is easy: a low glucose now predicts a low glucose next hour.
- **The time split is slightly higher than GroupKFold** on all datasets except ReplayBG GRU (0.809 vs 0.827). The model has already seen the earlier hours of the same person,
so the future of a known patient is a bit easier than a new patient.
The gap is small: 0.01–0.03 for most models, 0.065 for SVC on HUPA.
The ranking is the same in both protocols, with the Transformer >= GRU > SVC > LR on HUPA.

## Cost

One fit per dataset and model, on the time split.

- train_s is the full fit.
- epoch_s = train_s / epochs.
- peak_mb is process RSS just after the fit. It is not isolated per model.
- infer_ms is milliseconds per test window, at batch 64, after a warmup batch.

| Dataset | Model | train_s | epoch_s | peak_mb | infer_ms | AUPRC (time) |
|---|---|---:|---:|---:|---:|---:|
| simglucose | LR | 0.10 | — | — | 0.017 | 0.930 |
| simglucose | SVC | 66.34 | — | — | 0.761 | 0.931 |
| simglucose | GRU | 58.93 | 1.964 | 187.5 | 0.124 | 0.948 |
| simglucose | Transformer | 27.81 | 2.781 | 375.1 | 0.063 | 0.943 |
| HUPA-UCM | LR | 0.04 | — | — | 0.007 | 0.711 |
| HUPA-UCM | SVC | 52.14 | — | — | 0.714 | 0.808 |
| HUPA-UCM | GRU | 50.55 | 1.685 | 347.7 | 0.050 | 0.854 |
| HUPA-UCM | Transformer | 24.18 | 2.418 | 372.9 | 0.057 | 0.880 |
| ReplayBG | LR | 0.02 | — | — | 0.010 | 0.786 |
| ReplayBG | SVC | 2.47 | — | — | 0.127 | 0.811 |
| ReplayBG | GRU | 12.65 | 0.422 | 356.0 | 0.032 | 0.809 |
| ReplayBG | Transformer | 5.91 | 0.591 | 370.2 | 0.053 | 0.854 |


## What to do next

On simglucose, drop one channel at a time on single windows and check whether the score 
falls on the cause the simulator already knows: low glucose, insulin, or carbs.




