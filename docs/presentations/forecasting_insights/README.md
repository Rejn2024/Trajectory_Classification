# Forecasting trained pilots from a tactical observer's view

For the expanded original-versus-improved catalogue analysis, use the
[21-slide skills comparison and full metric report](../skills_catalogue/README.md).
This earlier 13-slide deck remains the combined motion/skills overview.

**13 slides, evidence snapshot 2 November 2026.**

[PDF slides](forecasting_insights.pdf) · [Offline HTML slides](index.html) ·
[Speaker notes](speaker_notes.md) · [Individual PNG slides](figures/)

Download the PDF, or use **Download raw file** for the HTML and open it locally.
Both are standalone and need no Python, network, model weights or Git LFS.
The HTML supports arrow keys, Home/End, **N** for notes and **F** for fullscreen.
The PNGs can be inserted into another slide deck. The PDF contains vector charts
and selectable text; the HTML embeds the rendered slides and supplementary notes.

This deck adds the initial skill forecasts and subsequent accuracy improvements
to the earlier motion findings. The [original presentation](../README.md), its
results and all local model artifacts are retained.

## The viewpoint and the two tasks

Scripted tactical aircraft **B01 observes trained aircraft A01**. The ten trained
policies are frozen during recording. Separate supervised models learn from B01's
observation histories to predict A01 at **+1, +2 and +5 seconds**:

- **Motion:** future position and velocity.
- **Catalogue selection:** the exact skill entry selected at the future time.

These forecasts do not rate pilot competence. Several catalogue entries share
physical guidance implementations, so selected labels need not identify distinct
executed manoeuvres. No pilot ID, private selection or future observation is an
inference input. The observations are idealised simulator tracks.

## Main findings

Motion results use equal episode weight: compute mean 3D error within each
episode, then average across 450 validation episodes. At +5 seconds, the original
transformer's position error is **19.09 m**, compared with **63.94 m** for constant
acceleration and **221.46 m** for constant velocity. The GRU leads short-horizon
position error; the transformer leads position and velocity at +5 seconds.

Skill results give equal weight to valid forecast origins:

| Horizon | Original transformer top-1 | Improved ensemble top-1 | Gain | Paired 95% interval for gain |
| --- | ---: | ---: | ---: | ---: |
| +1 s | 58.97% | 65.62% | +6.65 points | +5.60 to +7.84 points |
| +2 s | 58.38% | 65.17% | +6.79 points | +5.69 to +8.04 points |
| +5 s | 55.90% | 63.48% | +7.58 points | +6.39 to +8.95 points |

Top-1 improves for all ten pilots at all three horizons. Overall top-3 accuracy,
catalogue macro-F1 and negative log-likelihood also improve. On the subset where
the future selection differs from the current one, +5-second accuracy increases
from **40.18% to 55.33%**. Endpoint differences do not identify every intermediate
switch, and subset membership differs across horizons.

Not every subset improves. For example, +5-second unchanged-selection top-3
accuracy falls from 91.12% to 90.81%, while its log-loss rises from 1.0575 to 1.0813.
The deck's notes retain this qualification.

## Interpretation and limitations

The corpus contains **3,000 episodes**, representing **300 scenario groups × ten
pilots**. All ten versions of a scenario stay in the same partition: 210 training,
45 validation and 45 reserved test groups. The new model search uses two internal
partitions of the training groups, each with 147 fit / 21 stop / 42 selection groups.
The architecture, epoch counts, temperatures and seeds were fixed before the final
validation comparison.

The selected ensemble has **12 models**, combining four temporal MLP variants
with three seeds each, 91 observable/derived features and up to **60 seconds of
history**. The original transformer used five seconds. This larger history and
compute budget is part of the comparison. The experiments do not isolate the
individual contribution of every change. A six-model alternative had nearly
identical internal accuracy; twelve models are not established as necessary.

Bootstrap intervals resample the 45 complete validation scenario groups 5,000
times, keeping pilots and overlapping windows together. These are **development
intervals**, not a correction for all historical model selection. The original
validation set had already informed the initial models. **The reserved test set
remains unevaluated.** Unseen-pilot generalisation, noisy or missing observations,
and operational deployment have not been demonstrated.

## Slide map

| Slides | Content |
| --- | --- |
| 1–3 | Main findings, B01/A01 information flow and scenario partitions |
| 4–5 | Motion results and the meaning of catalogue forecasting |
| 6–9 | Exact-label accuracy, changed selections, every pilot and probability metrics |
| 10–11 | Model changes, compute/history trade-offs and training-only selection |
| 12–13 | Paired uncertainty, evidence limits and proposed next work |

## Evidence and rebuilding

The committed `data/` bundle contains the original and improved CSV metrics,
internal confirmation results, scenario bootstrap intervals, audit summaries and
SHA-256 hashes of the source artifacts. `data/checksums.json` protects the portable
inputs; `checksums.json` inventories this presentation. All packaged text uses LF
line endings so the checksums remain stable across Git checkouts.

Rebuild using the existing project's plotting dependencies, without the original
training artifacts, model checkpoints or corpus:

```powershell
.venv\Scripts\python.exe -X utf8 scripts/build_forecasting_insights.py
```

To deliberately re-collect and audit the saved local experiment results first:

```powershell
.venv\Scripts\python.exe -X utf8 scripts/build_forecasting_insights.py --collect --snapshot-date 2026-11-02
```

Neither command trains, simulates, runs model inference or evaluates the test
partition. The collection step checks the frozen experiment source hashes, exact
reload verification, training-only normalization, causal-history audits and all
76 preserved initial artifact hashes. The renderer checks the evidence checksums,
matched skill counts, bootstrap gain arithmetic and slide text bounds. A local
contact sheet is saved under `artifacts/checks/forecasting_insights/` for review.

Technical detail: [motion experiment](../../motion_forecasting.md),
[initial skill experiment](../../skill_forecasting.md),
[improvement study](../../skill_forecasting_improvements.md).
