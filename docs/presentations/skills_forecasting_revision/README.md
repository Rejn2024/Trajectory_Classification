# Skills forecasting: revision edition

**22 slides, four main model milestones, one revision question per slide.**

[Download the PDF](skills_forecasting_revision.pdf) · [Offline HTML deck](index.html) · [Study guide](study_guide.md) · [Speaker notes](speaker_notes.md)

This is the current teaching edition, dated **4 November 2026**. It brings the original experiment and the v3, v4 and v5 studies into one learning sequence. It uses fixed published evidence; building it does not train models, run inference or evaluate the reserved test.

The comparison follows four main references: original Transformer, twelve temporal MLPs (v3), the fifteen-member MLP/TCN reference (v4), and six pure Transformers (v5). Full candidate searches remain in the research reports. The secondary 21-member blend is discussed once on the uncertainty slide.

The main finding remains: v4 averages **66.19%** validation accuracy and the pure Transformers **65.88%**. The Transformers improve probability quality and use fewer parameters; their accuracy difference interval crosses zero. This is neither proof of an accuracy improvement nor equivalence.

## Use the deck for revision

Start at slide 1 and progress in order. Each slide has a check question; notes and the study guide explain the answer. The HTML is self-contained and supports arrows, Home/End, N for notes and F for fullscreen. The PDF contains vector diagrams and selectable text.

| Slide | Topic | Main idea |
|---:|---|---|
| 01 | What are we trying to forecast? | We predict a trained pilot's future catalogue choice from the tactical observer's available history. |
| 02 | Two different systems learn two different jobs | B01 observes A01 after the trained pilot has been frozen; forecasting does not update that pilot. |
| 03 | Why 3,000 encounters after 112,500 training flights? | The forecasting corpus contains 300 shared starting scenarios, each flown by all ten frozen pilots. |
| 04 | History, forecast horizon and elapsed time | A longer backward history is legitimate when every input was available at the forecast origin. |
| 05 | What observations does the forecaster actually use? | The 91 features describe observed state, relative motion, energy, missiles and past changes. |
| 06 | A 3D history can become an ordinary feature row | Flattening keeps each feature in a fixed time slot; it does not erase temporal ordering. |
| 07 | The temporal MLP and the twelve-model ensemble | The v3 ensemble combines four MLP recipes, each trained with three fixed random seeds. |
| 08 | The TCN adds a different view of temporal patterns | V4 averages 50% of the twelve-MLP mean with 50% of the three-TCN mean. |
| 09 | What the six new Transformers do | Attention combines observed history; the classification head directly predicts all three future horizons. |
| 10 | Four milestones explain the experiment history | The experiments changed information representation, training choices and averaging as well as architecture. |
| 11 | Which data can influence each training decision? | Later recipes were selected within original training, frozen, then evaluated on development validation. |
| 12 | Cross-entropy, NLL and the learning curves | Read the training objective, uncalibrated stop NLL and calibrated final NLL as distinct measurements. |
| 13 | Accuracy, precision, recall and F1 | Accuracy rewards the right top label; macro-F1 gives every catalogue label an equal place in the average. |
| 14 | How the main models compare on accuracy | V4 leads the four-model validation comparison; v5 Transformers did not improve its measured accuracy. |
| 15 | Accuracy is only one part of the scorecard | The smaller Transformer ensemble improves probability scores while losing some top-one accuracy and macro-F1. |
| 16 | Which trained pilots are easier to forecast? | Population averages hide substantial pilot differences and genuine regressions. |
| 17 | Per-skill F1 needs its support count | A large change for a rare label can be driven by very few correlated forecast origins. |
| 18 | What the temporal statistics actually tell us | Later-origin accuracy and label persistence describe the data; they do not prove that longer history caused a gain. |
| 19 | How certain are the measured differences? | The latest Transformer intervals cross zero; neither improved accuracy nor equivalence is established. |
| 20 | How much experiment work has been done? | Training effort is recorded per study; summed fit time is not project wall time or inference latency. |
| 21 | What remains uncertain, and what could improve? | Further improvement requires a new frozen training-only experiment with the same primary comparison task. |
| 22 | Revision recap: what can we now explain? | V4 leads measured accuracy; the smaller Transformer challenger improves probability quality, with important limits. |

## Evidence and previous editions

- [Compact evidence and source provenance](data/evidence.json), [main metrics](data/metrics.csv), [per-skill F1](data/skills.csv), [pilot intervals](data/pilot_intervals.csv).
- [Learning curves](data/learning_curves.csv), [temporal analysis](data/temporal.csv), [study costs](data/costs.csv), [paired comparisons](data/comparisons.csv), [checksums](checksums.json).
- [Historical 41-slide catalogue deck](../skills_catalogue/README.md), including the per-pilot RL-return curves and full original comparisons.
- [v4 sequence-model experiment](../../experiments/skill_forecasting_v4/README.md) and [v5 Transformer experiment](../../experiments/skill_forecasting_v5_transformers/README.md), with full configurations, searches and preserved results.

The historical decks and experiment bundles remain unchanged. The new edition replaces them as the main revision route, while retaining their detailed evidence. All figures describe known pilots under new scenario groups and idealised simulator observations. The validation set has been inspected during development; the reserved test remains unevaluated.

## Rebuild

```powershell
.venv\Scripts\python.exe -X utf8 scripts/build_skills_forecasting_revision.py

# Refresh the compact snapshot from the committed, fixed evidence bundles:
.venv\Scripts\python.exe -X utf8 scripts/build_skills_forecasting_revision.py --collect
```

The default rebuild uses only this edition's portable data snapshot and the tracked builder/content/helper scripts. It does not need checkpoints, raw observations or network access.
