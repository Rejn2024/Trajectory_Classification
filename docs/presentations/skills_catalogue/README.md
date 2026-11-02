# Skills catalogue forecasting: original experiments versus improved ensemble

**21 slides dedicated to the catalogue task**, supported by a complete metric
comparison and reproducible evidence from the experiments performed so far.

**Full slide pack on GitHub:**
[Download the 21-page PDF](https://github.com/Rejn2024/Trajectory_Classification/raw/refs/heads/forecasting/docs/presentations/skills_catalogue/skills_catalogue.pdf)
· [Open its GitHub file page](https://github.com/Rejn2024/Trajectory_Classification/blob/forecasting/docs/presentations/skills_catalogue/skills_catalogue.pdf).
This is the complete presentation on the `forecasting` branch.

[PDF slides](skills_catalogue.pdf) · [Offline HTML slides](index.html) ·
[Full comparison and metric definitions](comparison.md) ·
[Speaker notes](speaker_notes.md) · [Individual slide images](figures/)

Download the PDF, or download the HTML as a raw file and open it locally.
Both are standalone: no network, model weights, Python or Git LFS is needed to
view them. The HTML supports arrow keys, Home/End, **N** for notes and **F** for
fullscreen. The PDF has vector charts and selectable text; the PNGs can be reused
in another presentation. The [earlier motion/skills overview](../forecasting_insights/README.md)
is preserved.

## What the comparison measures

Scripted tactical aircraft **B01 observes trained aircraft A01**. The forecaster
uses B01's available history to predict the **exact catalogue selection at +1,
+2 and +5 seconds**. It predicts a selected label, not a rating of pilot competence.
Several catalogue entries can share executed guidance, so these are not
necessarily distinct physical manoeuvres.

The deck compares four methods throughout: **training frequency**, **original
GRU**, **original transformer**, and the **improved 12-model ensemble**. Each
horizon uses the same validation origins, labels and masks for every method.
The expanded analysis reuses saved predictions; it does not train, recalibrate,
change the selected model or evaluate the reserved test.

## Original transformer → improved ensemble

| Metric | +1 second | +2 seconds | +5 seconds |
| --- | ---: | ---: | ---: |
| Top-1 accuracy ↑ | 58.97% → 65.62% | 58.38% → 65.17% | 55.90% → 63.48% |
| Top-3 accuracy ↑ | 88.40% → 90.10% | 88.22% → 89.82% | 86.51% → 89.25% |
| Macro-F1, all 33 ↑ | 0.3010 → 0.4120 | 0.2953 → 0.4115 | 0.2739 → 0.3665 |
| Negative log-likelihood ↓ | 1.2216 → 1.1155 | 1.2361 → 1.1324 | 1.2903 → 1.1642 |
| Multiclass Brier score ↓ | 0.5552 → 0.4694 | 0.5588 → 0.4738 | 0.5830 → 0.4916 |
| Calibration error, ECE points ↓ | 8.75 → 2.94 | 9.11 → 2.93 | 8.74 → 3.42 |

The [full comparison](comparison.md) provides scorecards for **all four methods**,
including supported macro-F1, and defines every metric and averaging convention.
The accuracy gains are **6.65 / 6.79 / 7.58 percentage points**. Those correspond
to **16.20% / 16.32% / 17.19% relative reductions in top-1 error**, respectively.
These quantities have different denominators and should not be interchanged.

## What the headline gain hides

- At +5 seconds, the ensemble corrects **4,144** original-transformer errors and
  introduces **1,911** new errors. The net is **2,233 additional correct forecasts**;
  **8,847** origins remain wrong under both models. These are overlapping forecast
  windows, not independent encounters.
- All ten pilots improve at all three horizons, but +5-second gains range from
  **0.77 to 12.86 points**. Absolute accuracy still differs between pilots.
- Changed-selection +5-second top-1 rises **40.18% → 55.33%**. Unchanged-selection
  top-1 also improves, but its top-3 falls **91.12% → 90.81%** and its NLL worsens
  **1.0575 → 1.0813**. Endpoint subsets are diagnostic; they are not switch forecasts.
- Eleven labels account for **96.42%** of +5-second truth support. Fifteen more
  labels have fewer than 500 occurrences, and seven have no truth support. The
  class slides use support-based groups, show every supported entry, and retain
  all 33 labels in the reported metrics.
- The ensemble improves NLL, Brier and ECE over both original neural models.
  However, the frequency baseline has very low ECE despite poor accuracy. A
  near-constant confidence can match its aggregate hit rate without distinguishing
  examples. Calibration error should be read alongside accuracy and proper
  probability scores.

## What changed in the experiments

The original GRU and transformer use **five seconds of history**, **54 features**,
one seed and **234,723 / 439,651 parameters**, respectively. They select epochs
15 / 20 using validation NLL. The improved ensemble uses **20- and 60-second
histories**, **91 features**, four temporal MLP variants and three seeds per variant,
with **28,977,060 total parameters** and equal probability averaging.

New model selection uses two partitions inside the original 210 training groups:
147 groups for fitting and normalization, 21 for stopping/calibration, and 42 for
candidate comparison. Final components, epochs, temperatures and seeds are fixed
before the outer validation comparison. The slide deck includes unsuccessful
initial candidates as well as the selected result. These experiments change
several factors together and do not isolate the effect of every change.

A smaller fixed mixture had nearly identical internal accuracy (**60.96% versus
60.98%**). The evidence does not establish that twelve final models are necessary.
The larger history and compute budget are explicit trade-offs, not a free
architecture improvement.

## Slide map

| Slides | Comparison |
| --- | --- |
| 1–4 | Task, shared evaluation population, original versus improved setup, metric definitions |
| 5–7 | Complete four-method scorecards at +1, +2 and +5 seconds |
| 8–9 | Percentage-point gains, relative error reduction, corrections and regressions |
| 10–11 | Changed/unchanged subsets and absolute per-pilot accuracy |
| 12–15 | Class imbalance, frequent/sparse-label F1 and paired confusion matrices |
| 16–17 | Brier, calibration error and reliability diagrams |
| 18–19 | Successful and unsuccessful development choices, selection procedure and cost |
| 20–21 | Scenario uncertainty, limitations and conclusions |

## Limits of the evidence

The corpus has 300 scenario groups × ten known pilots, split 210 / 45 / 45 into
training, validation and reserved test. All pilots from a scenario stay together.
The reported paired intervals resample all 45 validation scenario groups 5,000
times. They remain positive at all horizons, but describe **development evidence**:
the original validation set informed the original models, and internal model
selection was adaptive. They do not correct for all historical model selection.

**The reserved test remains unevaluated.** Unseen-pilot generalisation and noisy
or missing sensor observations have not been demonstrated. Rare-label estimates
are unstable; absent-label performance is not measured. The new Brier/ECE and
paired-error analyses are descriptive and have not been used to retune anything.

## Portable evidence and rebuilding

The `data/` directory contains all four models' per-horizon, per-pilot and cohort
metrics; all 33 classes' precision/recall/F1 at every horizon; ungrouped confusion
counts; reliability bins and their sample counts; paired corrections/regressions;
internal experiment results; and scenario intervals. Source SHA-256 hashes record
the fixed original artifacts. The collector independently reproduces all twelve
model/horizon scorecards and confusion matrices from the saved probabilities.

```powershell
# Rebuild from committed evidence; no original checkpoints or dataset needed.
.venv\Scripts\python.exe -X utf8 scripts/build_skill_catalogue_presentation.py

# Deliberately audit the local saved predictions and regenerate the evidence bundle.
.venv\Scripts\python.exe -X utf8 scripts/build_skill_catalogue_presentation.py --collect
```

Both commands use the project's existing plotting/data dependencies. The first
reads only the portable bundle. The second reads fixed validation predictions,
never runs model inference and never reads test targets. Unit checks cover the
new metric formulas and paired-error accounting. All text is normalized to LF
for stable Git checksums. The original data, predictions and model weights remain
unchanged.
