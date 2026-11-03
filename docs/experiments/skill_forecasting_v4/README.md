# Skill forecasting v4: architecture experiment

The frozen v4 recipe changes mean accuracy by **+1.44 percentage points** against the retained v3 ensemble (paired scenario-bootstrap 95% interval **[+0.90, +2.08] pp**).

[Open the five-page PDF](skill_forecasting_v4.pdf) · [Speaker notes](speaker_notes.md)

The comparator is the current **12-member v3 MLP ensemble**. Both versions use the same ten trained pilots, 33 catalogue labels, exact +1/+2/+5-second targets, forecast origins and scenario partitions. B01 is the tactical observer of the frozen trained pilots. Labels describe selected catalogue behavior, not competence.

| Horizon | Model | Top-1 | Top-3 | Macro-F1 / 33 | NLL | Brier | ECE |
|---|---|---:|---:|---:|---:|---:|---:|
| +1 s | Retained v3 | 65.62% | 90.10% | 0.4120 | 1.1155 | 0.4694 | 2.94 pp |
| +2 s | Retained v3 | 65.17% | 89.82% | 0.4115 | 1.1324 | 0.4738 | 2.93 pp |
| +5 s | Retained v3 | 63.48% | 89.25% | 0.3665 | 1.1642 | 0.4916 | 3.42 pp |
| +1 s | Frozen v4 | 67.09% | 91.01% | 0.4125 | 1.0688 | 0.4515 | 4.16 pp |
| +2 s | Frozen v4 | 66.62% | 90.78% | 0.4180 | 1.0855 | 0.4564 | 3.63 pp |
| +5 s | Frozen v4 | 64.86% | 89.86% | 0.3767 | 1.1243 | 0.4763 | 4.36 pp |

ECE worsened at all three horizons (2.94/2.93/3.42 → 4.16/3.63/4.36 pp at +1/+2/+5 s); NLL and Brier improved.

Selected budget: **15 models / 31.73M parameters**. This experiment completed 37 new fits, with 57.09 minutes of summed logged training time; loading and setup are excluded. Some confirmation fits ran concurrently on one shared GPU: their timers include resource contention and do not measure end-to-end elapsed time or isolated-model throughput.

Selection used two grouped partitions entirely within original training. Epochs and temperature came from stop groups; recipe selection used selection groups. The final recipe was frozen before any new outer-validation evaluation. Those internal partitions overlap and have guided previous development. Outer validation has also been inspected previously. The reserved test remains untouched; these are development results for new scenarios with known pilots.

Expanded blends increase model count. Standalone and blended scores are retained, but this comparison does not isolate an architecture-only causal gain. Bootstrap intervals resample whole shared scenario groups and are pointwise, without multiple-comparison correction.

## Portable evidence

- [Full metrics](data/metrics.csv): population and pilot metrics at each horizon, including NLL, Brier and ten-bin ECE.
- [Pilot comparisons](data/pilot_comparison.csv): support and paired scenario intervals.
- [Pilot × skill metrics](data/pilot_skill_metrics.csv) and [skill support](data/skill_support.csv): P/R/F1, counts and coverage.
- [Internal screen](data/sequence_screen.csv), [confirmation](data/sequence_confirmation.csv) and [frozen selection](data/frozen_selection.json).
- [Run inventory](data/run_inventory.csv) and [training histories](data/training_histories.csv): completed/failed runs, seed, epoch, timing and parameters.
- [Common-origin temporal results](data/temporal_common_origin_metrics.csv) and [accuracy weighting comparisons](data/accuracy_weightings.csv).
- [Verification](data/verification.json), [provenance](data/evidence.json) and [bundle checksums](checksums.json).

## Exact experiment provenance

[frozen_protocol.json](data/frozen_protocol.json) preserves the original protocol bytes and its SHA-256 matches the hash recorded in the frozen selection. [protocol.json](data/protocol.json) is a convenience copy with only the data and baseline directory paths rewritten as repository-relative paths; its bytes intentionally have a different hash. [provenance.json](data/provenance.json) records both hashes and the source-archive hash.

The [preservation manifest](data/preserved_artifacts.json) contains every original dataset/experiment file hash. The [source snapshot ZIP](source_snapshot.zip) contains the exact registered training source, configuration and CLI bytes at repository-relative paths, with fixed ZIP timestamps. It includes no checkpoints or datasets. Extract it into an isolated repository checkout when exact source replay is required; avoid line-ending conversion, which changes registered hashes even when Python behavior is unchanged. The original source hashes remain inside both protocol copies.

## Rebuild

From the repository root, the portable PDF can be rebuilt without simulator data or checkpoints:

```powershell
.venv\Scripts\python.exe scripts/report_sequence_experiments.py
```

Collect the completed local experiment and regenerate the report:

```powershell
.venv\Scripts\python.exe scripts/report_sequence_experiments.py --collect
```

Reproduce training only in a **new, empty output directory**; preserve existing runs:

```powershell
.venv\Scripts\python.exe scripts/sequence_skill_forecaster.py initialize --output artifacts/skill_forecasting/v4_reproduction
.venv\Scripts\python.exe scripts/sequence_skill_forecaster.py screen --output artifacts/skill_forecasting/v4_reproduction
.venv\Scripts\python.exe scripts/sequence_skill_forecaster.py confirm --output artifacts/skill_forecasting/v4_reproduction
.venv\Scripts\python.exe scripts/sequence_skill_forecaster.py refit --output artifacts/skill_forecasting/v4_reproduction
.venv\Scripts\python.exe scripts/sequence_skill_forecaster.py evaluate --output artifacts/skill_forecasting/v4_reproduction
```

The final evaluation command refuses a second evaluation in that output directory. Reproduction requires the preserved local dataset and baseline artifacts, listed by hash in the protocol. GPU/software differences may change numerical results. The earlier 41-slide catalogue presentation is preserved separately.
