# Skill forecasting v5: Transformer challenge

[Open the six-page PDF](skill_forecasting_v5_transformers.pdf) · [Speaker notes](speaker_notes.md)

The primary question is whether an all-Transformer challenger can match the retained **15-model v4 MLP/TCN ensemble**. The primary result is reported even if worse. Any secondary incumbent blend is a separate model family and does not establish a Transformer-only replacement.

Transformer only: mean accuracy gain -0.31 pp; 95% scenario interval [-1.18, +0.64] pp.

Secondary blend: mean accuracy gain +0.07 pp; 95% scenario interval [-0.63, +0.84] pp.

These are the **best tested recipes**, not a demonstrated optimum. The same 33 labels, ten trained pilots, forecast origins, target masks and +1/+2/+5-second horizons are retained. No equivalence margin was registered: a confidence interval crossing zero does not prove equal performance.

| Horizon | Model | Top-1 | Top-3 | Macro-F1 / 33 | NLL | Brier | ECE |
|---|---|---:|---:|---:|---:|---:|---:|
| +1 s | v4 incumbent | 67.09% | 91.01% | 0.4125 | 1.0688 | 0.4515 | 4.16 pp |
| +1 s | Transformer only | 66.67% | 91.23% | 0.4119 | 1.0377 | 0.4484 | 2.63 pp |
| +1 s | Secondary blend | 67.10% | 91.47% | 0.4171 | 1.0227 | 0.4439 | 3.51 pp |
| +2 s | v4 incumbent | 66.62% | 90.78% | 0.4180 | 1.0855 | 0.4564 | 3.63 pp |
| +2 s | Transformer only | 66.47% | 90.98% | 0.4130 | 1.0497 | 0.4515 | 2.34 pp |
| +2 s | Secondary blend | 66.73% | 91.20% | 0.4169 | 1.0353 | 0.4474 | 2.94 pp |
| +5 s | v4 incumbent | 64.86% | 89.86% | 0.3767 | 1.1243 | 0.4763 | 4.36 pp |
| +5 s | Transformer only | 64.50% | 90.14% | 0.3639 | 1.0980 | 0.4744 | 3.82 pp |
| +5 s | Secondary blend | 64.96% | 90.39% | 0.3680 | 1.0852 | 0.4699 | 4.71 pp |

Transformer only: NLL improved at all horizons; Brier improved at all horizons; ECE improved at all horizons.

Primary: 60 s patch T + 60 s temporal T; 6 Transformer members, no incumbent members.

Secondary blend: NLL improved at all horizons; Brier improved at all horizons; ECE changed across horizons.

Secondary: 75% new (60 s patch T + 60 s temporal T) + 25% preserved v4; 21 total models.

The **preliminary screen compares individual candidate models** with the cached **15-member v4 fold-0 reference**, which has mean accuracy **60.5334%**, read from [internal_baseline_metrics.json](data/internal_baseline_metrics.json). The final primary is a **multi-seed ensemble**: three seeds per shortlisted architecture, confirmed on both internal partitions.

Each partition versus v4: NLL <= v4 +0.02; mean F1/33 >= v4 -0.01; each horizon accuracy >= v4 -0.003 (-0.30 pp). Secondary: mean accuracy gain >0 in both partitions. F1/33 means the mean of the three all-33-label macro-F1 scores. The primary is retained as a labelled diagnostic challenger if no pure recipe passes all guards; a secondary blend must pass the guards and the positive-gain requirement.


Transformer-family models retain standard feed-forward blocks and current-state readouts. The explicitly labelled architectural hybrid adds a parallel full-history MLP. Attention connects only observations supplied at or before the forecast origin; fixed time/gap coordinates describe the existing sampling layout.

The study completed **29 new fits**, reusing 30 internal baseline fits. Summed recorded fit time was **135.30 minutes**, excluding loading/setup. Any concurrent GPU work introduces contention; this is not end-to-end elapsed time or an isolated throughput benchmark.

Reported outputs were frozen before one outer-validation invocation. All selection used grouped partitions inside original training. Those partitions overlap and have guided earlier studies; outer validation has also been inspected before. These are development results for new scenarios involving known pilots, not an untouched final test or unseen-pilot evaluation. The reserved test remains untouched.

## Complete evidence

- [All population and pilot metrics](data/metrics.csv), [pilot comparisons and intervals](data/pilot_comparison.csv), [probability reliability](data/reliability.csv).
- [All pilot/skill precision, recall and F1](data/pilot_skill_metrics.csv), [skill support and coverage](data/skill_support.csv).
- [Screen](data/transformer_screen.csv), [confirmation](data/transformer_confirmation.csv), [frozen recipes](data/frozen_selection.json).
- [Run inventory](data/run_inventory.csv), [training histories](data/training_histories.csv), [temporal comparisons](data/temporal_common_origin_metrics.csv).
- [Joint mean intervals](data/mean_horizon_intervals.json), [per-horizon intervals](data/paired_scenario_bootstrap.csv), [alternative accuracy weightings](data/accuracy_weightings.csv).
- [Verification](data/verification.json), [source provenance](data/evidence.json), [bundle checksums](checksums.json).

## Exact provenance and reproduction

[frozen_protocol.json](data/frozen_protocol.json) contains the byte-exact registered protocol. Its hash matches the frozen selection. [protocol.json](data/protocol.json) only normalizes data/baseline paths for readability and has a different hash. [provenance.json](data/provenance.json) records both hashes. Keep the frozen protocol and preservation manifest without line-ending conversion.

The [preservation manifest](data/preserved_artifacts.json) records all earlier artifacts. The deterministic [source archive](source_snapshot.zip) contains exact registered code/configuration/CLI bytes, without datasets or checkpoints. Extract it into an isolated checkout when reproducing the frozen source. Earlier PDFs and results remain unchanged.

Rebuild the portable PDF without raw observations or model checkpoints:

```powershell
.venv\Scripts\python.exe scripts/report_transformer_experiments.py
```

Collect a completed local experiment and rebuild:

```powershell
.venv\Scripts\python.exe scripts/report_transformer_experiments.py --collect
```

A new training reproduction requires preserved prepared data and baseline artifacts, and a fresh output directory. Use `scripts/transformer_skill_forecaster.py` commands `initialize`, `screen`, `confirm`, `refit`, and `evaluate` with that output. Do not overwrite this experiment or select another recipe using its validation scores. Hardware/software differences may alter numerical results.
