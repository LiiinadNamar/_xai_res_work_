**Goal.**  
Three things:
1. Add k-fold (GroupKFold) so each model is trained and tested on separate patients
— the whole patient is either in train or in test, never both.
The intent is the same spirit as "train on known data, validate on unseen":
no information from a test patient leaks into training, and the score is averaged over 5 folds instead of one lucky split.
2. Add a Transformer as a separate sequence model.
Unlike the GRU, which reads the hour step by step, self-attention looks at the
whole 12-step window at once and learns by itself which moments matter.
3. Simulate new, full datasets that resemble real-life CGM data,
so the comparison is not limited to tiny cached samples: a large simglucose set and a ReplayBG digital-twin set built on real HUPA days,
next to the real HUPA-UCM cohort.
4. Add training profile: monitor models' training time per epoch, memory required, inference time

**What I did (short).**  
Built hourly windows from 5-minute tables, labelled hypo in the next hour, and compared four models — LR, SVC, GRU, 
and a tiny Transformer — under one shared GroupKFold(5) on patient. 
Reported AUPRC mean ± std, plus training time, memory and per-epoch timing. 
Fixed seeds including batch-shuffle order for the neural nets.

## Windows, features and label

- All datasets are on a 5-min grid with glucose, insulin per bin (U) and carbohydrates per bin (g).
- One window per hour (STEP = 12). Each window uses the past hour (LOOKBACK = 12 bins) and is labelled from the next hour (HORIZON = 12 bins).
- Windows with any missing value in the past or future hour are dropped.
- **Label** hypo = 1 if the minimum glucose in the next 12 bins is below 70 mg/dL.
- **Tabular features (LR, SVC):** glucose_mean, glucose_min, glucose_slope (last minus first value, per hour), insulin_sum, carb_sum.
- **Sequence input (GRU, Transformer):** the raw 12 × 3 past-hour sequence (glucose, insulin, carbohydrates), from the same windows as the tabular features.

**Datasets (full simulated + real).**

| Dataset | Source | Windows | Patients | Window hypo rate |
|---|---|---:|---:|---:|
| simglucose | simulator | 28,868 | 28 | 31.6% |
| HUPA-UCM | real CGM | 25,746 | 25 | 11.6% |
| ReplayBG | digital twin on HUPA days | 17,811 | 21 | 13.4% |

## Models

| Model | Input | Configuration |
|---|---|---|
| LR | 5 features | StandardScaler + LogisticRegression(class_weight="balanced", random_state=0) |
| SVC | 5 features | RBF SVC probability=True (or LinearSVC + calibration on >30k train) |
| GRU | 12×3 sequence | 1×GRU(16) + linear head; 30 epochs; Adam 1e-3; batch 64; BCEWithLogitsLoss + pos_weight |
| Transformer (separate) | 12×3 sequence | Linear-32, sin/cos positional encoding, 1×TransformerEncoder (2 heads, d_ff 64), mean pool, linear - 1; 10 epochs; same loss |

## Protocol

- **GroupKFold(5) on patient for every dataset and every model.** All windows of one patient stay together. All four models use exactly the same train/test indices per fold.
- Scaling (and per-channel standardisation for the neural nets) is fit on the training fold only.
- If a model fails on a fold, the traceback is printed and the loop continues; in the executed run nothing failed.
- **Seeds.** np.random.seed(0) and torch.manual_seed(0) at start; random_state=0 for LR/SVC; torch.manual_seed(0) before each neural fit;
- **batch order fixed** with BATCH_SHUFFLE_SEED + epoch. Dataset-generation seeds live in generate-datasets.ipynb.

## Training profile

Mean over folds (peak_memory_mb = process peak RSS, not isolated GPU):

| Dataset | Model | Train (s) | Infer (s) | Epoch mean (s) |
|---|---|---:|---:|---:|
| simglucose | LR | 0.07 | ~0 | — |
| simglucose | SVC | 92.82 | 5.55 | — |
| simglucose | GRU | 86.14 | 0.05 | 2.87 |
| simglucose | Transformer | 41.24 | 0.23 | 4.12 |
| HUPA-UCM | LR | 0.05 | ~0 | — |
| HUPA-UCM | SVC | 66.08 | 3.24 | — |
| HUPA-UCM | GRU | 62.15 | 0.03 | 2.07 |
| HUPA-UCM | Transformer | 32.90 | 0.20 | 3.29 |
| ReplayBG | LR | 0.03 | ~0 | — |
| ReplayBG | SVC | 62.44 | 3.65 | — |
| ReplayBG | GRU | 43.00 | 0.02 | 1.43 |
| ReplayBG | Transformer | 20.65 | 0.14 | 2.07 |

## Results (executed run, 2026-11-17)

**AUPRC mean +- std (5-fold GroupKFold):**

| Model | simglucose | HUPA-UCM | ReplayBG |
|---|---:|---:|---:|
| LR | 0.917 ± 0.027 | 0.695 ± 0.132 | 0.373 ± 0.314 |
| SVC | 0.917 ± 0.019 | 0.743 ± 0.130 | 0.365 ± 0.301 |
| GRU | 0.938 ± 0.018 | 0.830 ± 0.097 | 0.364 ± 0.307 |
| Transformer | 0.932 ± 0.018 | 0.848 ± 0.089 | 0.365 ± 0.318 |

 **Reading the results**

On the real HUPA set — the Transformer is at least as good as the GRU and 
better than LR/SVC. So the idea that look at the whole window and pick important moments partly works 
for next-hour hypo, even with a very small encoder (1 layer, 2 heads). On simglucose everyone is already strong, 
so the Transformer does not change the story much. On ReplayBG it does not rescue the twin data either — 
which again points to the data/evaluation issue, not to a weak architecture.

GroupKFold vs the earlier single holdout (LR / SVC / GRU)

| | simglucose (old -> new) | HUPA-UCM (old → new) | ReplayBG (old -> new) |
|---|---|---|---|
| LR | 0.760 -> **0.917** | 0.657 -> **0.695** | 0.920 -> **0.373** |
| SVC | 0.728 -> **0.917** | 0.750 -> 0.743 | 0.883 -> **0.365** |
| GRU | 0.696 -> **0.938** | 0.837 -> 0.830 | 0.865 -> **0.364** |

- **simglucose jumps up** for all models. The old numbers came from a tiny cached sample (142 test windows, 2 virtual patients); the full simulated set has 28 patients and far more windows, so the CV estimate is higher and much more stable (std ~ 0.02).
- **HUPA-UCM barely moves** and keeps the same ranking — sequence models (now Transformer ~ GRU) still beat tabular ones. This is the trustworthy real-world signal: averaging over 5 patient folds gives essentially the same story as the single holdout, which is reassuring.
- **ReplayBG collapses** from ~0.9 to ~0.36.

### Why simulated ReplayBG now looks so bad

Before, I tested on the same patient. Now with GroupKFold we have 21 patients. 
I think the simulated signal does not generalise well from one twin to another, so the score drops a lot.
ReplayBG glucose is not a real sensor it is the model’s median CGM. Each twin has its own fitted parameters, 
so hypo patterns look different from patient to patient. That is why one fold can be ~0.91 and another ~0.12–0.31, and the mean ends up ~0.36 with a huge spread +-0.31.
So the ReplayBG number is really means smooth simulated twin does not give a hypo pattern that generalises across patients. 
The old high score was most probably an artefact of testing on the same patient's future.

---

**What to do next**

1. **Fix ReplayBG.** Make the twin data behave more like real CGM before trusting it.
2. **Save attention maps for the Transformer.** The current run does not store them — I did not think to save attention in this pass.
Next run: keep the encoder's attention weights and save a per-window heatmap over the 12 time steps, so we can see which minutes before a predicted hypo the model attends to.

  



