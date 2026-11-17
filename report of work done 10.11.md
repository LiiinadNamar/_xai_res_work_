**Goal.** 
Check that three simple models can predict hypoglycaemia in the next hour on three datasets, 
before any explanation methods.

**What I did.** 
From the past hour of glucose, insulin and carbs, each model predicts whether glucose goes below 70 in the next hour. 
Models: logistic regression, SVC, and a small GRU. Metric: AUPRC on the test set.

# Windows, features and label
- All datasets are on a 5min grid with glucose, insulin per bin (U) and carbohydrates per bin (g).
- One window per hour (STEP = 12). Each window uses the past hour (LOOKBACK = 12 bins) and is labelled from the next hour (HORIZON = 12 bins). Windows with any missing value in the past or the future hour are dropped.
- **Label** hypo = 1 if the minimum glucose in the next 12 bins is below 70.
- **Tabular features (LR, SVC):** glucose_mean, glucose_min, glucose_slope (last minus first value, per hour), insulin_sum, carb_sum.
- **Sequence input (GRU):** the raw 12 x 3 past-hour sequence (glucose, insulin, carbohydrates). It comes from the same windows as the tabular features.

# Protocol
- **simglucose and HUPA-UCM:** patient holdout with GroupShuffleSplit(test_size=0.3, random_state=0), so no patient appears in both train and test.
- **ReplayBG twin:** one patient only, so the split is chronological: the first 70% of windows train the model and the last 30% test it.
- All three models use exactly the same train and test windows for a dataset.
- Scaling (and per-channel standardisation for the GRU) is fitted on the training windows only.
- If a model fails on a dataset, the traceback is printed in the notebook and the other pairs still run. In the executed run, no pair failed.
Seeds. Seeds are set in the code: np.random.seed(0) and torch.manual_seed(0) at the start, random_state=0 for the split and for LR/SVC, torch.manual_seed(0) again before each GRU,
plus fixed seeds for simglucose and ReplayBG. With the same packages and the same machine, a full rerun should give the same AUPRC numbers.

# Results (executed run, 2026-11-10)
AUPRC on the test windows:

| Model | simglucose | HUPA-UCM | ReplayBG twin |
|---|---|---|---|
| LR | 0.760 | 0.657 | 0.920 |
| SVC | 0.728 | 0.750 | 0.883 |
| GRU | 0.696 | 0.837 | 0.865 |

Split sizes and class balance (the same for all models within a dataset):

| Dataset | Split | Train windows | Test windows | Train positives | Test positives | Test prevalence |
|---|---|---|---|---|---|---|
| simglucose | by patient | 142 | 142 | 55 | 42 | 0.296 |
| HUPA-UCM | by patient | 18,634 | 7,112 | 2,292 | 691 | 0.097 |
| ReplayBG twin | chronological | 234 | 101 | 70 | 26 | 0.257 |

Time in seconds, from the latest executed run: 
|---|
|simglucose - LR 0.0, SVC 0.0, GRU 2.3.|
|HUPA-UCM - LR 0.0, SVC 59.9, GRU 60.1. |
|ReplayBG - LR 0.0, SVC 0.0, GRU 0.7. |

These times move between runs. An earlier run on the same machine took 131.5 s for the HUPA-UCM SVC and 136.9 s for the GRU. The AUPRC values did not change.

**Reading the results.**
- All nine scores are well above their prevalence baselines, so all three model types learn the task on all three datasets.
- On HUPA-UCM, the only large real-world set, the ranking is GRU > SVC > LR (0.837, 0.750, 0.657). Non-linear models and the raw sequence seem to add information there.
- On the two small sets, LR scores highest. With 142 or 101 test windows, these differences are within what a different split or seed could change.

# Reproducibility checklist

**Environment** (macOS 15.7.9, Intel x86_64, project venv `.venv`):

| Package | Version |
|---|---|
| Python | 3.12.12 |
| numpy | 2.0.2 |
| pandas | 3.1.0 |
| scipy | 1.18.1 |
| scikit-learn | 1.9.1 |
| torch | 2.2.2 (CPU) |
| simglucose | 0.2.11 |
| py_replay_bg | 1.4.0 |
| matplotlib | 3.11.2 |
| nbconvert | 7.17.2 |

torch 2.2.2 is the last wheel for Intel macOS and is built against NumPy 1.x. Under NumPy 2 it imports with a warning, and its NumPy bridge does not work. The notebook silences the import warning and passes tensors as Python lists (`tolist()`). On Apple silicon or Linux, a newer torch works without this workaround.

**Seeds.**
- np.random.seed(0) and torch.manual_seed(0) at start; torch.manual_seed(0) again before each GRU fit.
- GroupShuffleSplit(random_state=0); random_state=0 for LR, SVC and LinearSVC.
- simglucose CGM sensor seeds 1 to 4 (one per patient).
- ReplayBG seed=1 and np.random.seed(1) before the MAP fit.
- The GRU on CPU is deterministic for a fixed torch version and thread count. Small differences are possible on other hardware.








