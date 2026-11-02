# Skills catalogue: original experiments versus improved ensemble

This comparison predicts the **exact A01 catalogue selection at +1, +2 or +5 seconds** from scripted tactical observer B01's observations. It does not rate pilot competence. All results are development validation on the same 45 scenario groups and ten known trained pilots.

## Experiments and information budgets

| Experiment | History / features | Parameters | Selection |
| --- | --- | ---: | --- |
| Training frequency | No observation history; per-horizon training priors | No neural model | Training targets only |
| Original GRU | 5 s / 54 | 234,723 | Seed 42, best epoch 15 by validation NLL |
| Original transformer | 5 s / 54 | 439,651 | Seed 42, best epoch 20 by validation NLL |
| Improved ensemble | 20/60 s / 91; 12 temporal MLPs | 28,977,060 | Two training-only grouped partitions; freeze before outer validation |

Both initial neural models use width 128, two layers, dropout 0.1 and AdamW. The improved model combines four variants and three seeds each, with equal probability weights. This comparison changes architecture, history and compute together; it is not a controlled single-factor ablation.

## Reading the metrics

- **Top-1 / top-3:** correct label is the first choice / among the three highest probabilities. Higher is better.
- **Macro-F1 (33):** equal mean of per-label F1 across all catalogue entries; undefined F1 is zero.
- **Supported macro-F1 (26):** same calculation over labels with validation truth support. This changes the averaging set, not the task vocabulary.
- **Precision / recall:** fraction of predictions of a label that are correct / fraction of its true occurrences recovered. F1 is their harmonic mean. Support is the number of true occurrences.
- **NLL:** mean negative log probability of the true label (natural logarithm), clipped at 1e-12. Lower is better.
- **Brier:** mean sum of squared probability errors over all 33 classes, without dividing by 33; range 0–2. Lower is better.
- **ECE:** weighted absolute confidence–accuracy gap in ten fixed equal-width top-label bins. Reported in percentage points. It is descriptive, bin-dependent and not a complete calibration assessment.
- **Percentage-point gain** is new accuracy minus old accuracy; **relative error reduction** divides that gain by old error rate.

## +1-second scorecard: 30,802 valid origins

| Model | Top-1 % | Top-3 % | F1 (33) | F1 (26) | NLL | Brier | ECE pts |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Training frequency | 17.84 | 50.03 | 0.0092 | 0.0116 | 2.3921 | 0.8874 | 0.49 |
| Original GRU | 54.40 | 86.95 | 0.2374 | 0.3013 | 1.2606 | 0.5856 | 6.88 |
| Original transformer | 58.97 | 88.40 | 0.3010 | 0.3820 | 1.2216 | 0.5552 | 8.75 |
| Improved ensemble | 65.62 | 90.10 | 0.4120 | 0.5230 | 1.1155 | 0.4694 | 2.94 |

Top-1 gain: **6.65 points**. Relative top-1 error reduction: **16.20%**.

## +2-second scorecard: 30,466 valid origins

| Model | Top-1 % | Top-3 % | F1 (33) | F1 (26) | NLL | Brier | ECE pts |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Training frequency | 17.99 | 50.41 | 0.0092 | 0.0117 | 2.3899 | 0.8866 | 0.48 |
| Original GRU | 54.01 | 86.72 | 0.2309 | 0.2931 | 1.2768 | 0.5912 | 6.99 |
| Original transformer | 58.38 | 88.22 | 0.2953 | 0.3748 | 1.2361 | 0.5588 | 9.11 |
| Improved ensemble | 65.17 | 89.82 | 0.4115 | 0.5223 | 1.1324 | 0.4738 | 2.93 |

Top-1 gain: **6.79 points**. Relative top-1 error reduction: **16.32%**.

## +5-second scorecard: 29,458 valid origins

| Model | Top-1 % | Top-3 % | F1 (33) | F1 (26) | NLL | Brier | ECE pts |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Training frequency | 18.46 | 51.48 | 0.0094 | 0.0120 | 2.3799 | 0.8841 | 0.42 |
| Original GRU | 51.56 | 85.04 | 0.2123 | 0.2694 | 1.3363 | 0.6123 | 6.75 |
| Original transformer | 55.90 | 86.51 | 0.2739 | 0.3476 | 1.2903 | 0.5830 | 8.74 |
| Improved ensemble | 63.48 | 89.25 | 0.3665 | 0.4651 | 1.1642 | 0.4916 | 3.42 |

Top-1 gain: **7.58 points**. Relative top-1 error reduction: **17.19%**.

## Corrections and regressions on the same origins

| horizon_s | count | both_correct | corrected | regressed | both_wrong | net_correct | accuracy_gain |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 30802 | 16243 | 3968 | 1920 | 8671 | 2048 | 0.06649 |
| 2 | 30466 | 15872 | 3983 | 1914 | 8697 | 2069 | 0.06791 |
| 5 | 29458 | 14556 | 4144 | 1911 | 8847 | 2233 | 0.0758 |

These are forecast windows, not independent episodes. Net gain equals corrected minus regressed. This is descriptive reuse of fixed predictions, not a new model-selection step.

## Changed and unchanged endpoint labels

### +1 seconds

| model | cohort | count | top1_accuracy | top3_accuracy | macro_f1_catalogue | nll |
| --- | --- | --- | --- | --- | --- | --- |
| gru | same | 28538 | 56.0691 | 88.3068 | 0.2513 | 1.2 |
| gru | changed | 2264 | 33.3481 | 69.788 | 0.1285 | 2.0251 |
| transformer | same | 28538 | 60.5754 | 89.4386 | 0.317 | 1.1614 |
| transformer | changed | 2264 | 38.6926 | 75.3092 | 0.1746 | 1.9798 |
| ensemble | same | 28538 | 67.0404 | 90.7141 | 0.4373 | 1.077 |
| ensemble | changed | 2264 | 47.659 | 82.4205 | 0.2776 | 1.6007 |

### +2 seconds

| model | cohort | count | top1_accuracy | top3_accuracy | macro_f1_catalogue | nll |
| --- | --- | --- | --- | --- | --- | --- |
| gru | same | 26212 | 57.0884 | 89.3141 | 0.2575 | 1.165 |
| gru | changed | 4254 | 35.0259 | 70.7569 | 0.1336 | 1.9657 |
| transformer | same | 26212 | 61.3803 | 90.1991 | 0.3252 | 1.1279 |
| transformer | changed | 4254 | 39.8919 | 76.0461 | 0.1847 | 1.9031 |
| ensemble | same | 26212 | 67.553 | 90.7828 | 0.4456 | 1.072 |
| ensemble | changed | 4254 | 50.4937 | 83.874 | 0.328 | 1.5049 |

### +5 seconds

| model | cohort | count | top1_accuracy | top3_accuracy | macro_f1_catalogue | nll |
| --- | --- | --- | --- | --- | --- | --- |
| gru | same | 20799 | 57.8105 | 90.9034 | 0.2655 | 1.1098 |
| gru | changed | 8659 | 36.5631 | 70.9435 | 0.1416 | 1.8804 |
| transformer | same | 20799 | 62.4453 | 91.1198 | 0.3352 | 1.0575 |
| transformer | changed | 8659 | 40.1778 | 75.4475 | 0.1945 | 1.8496 |
| ensemble | same | 20799 | 66.8734 | 90.8121 | 0.373 | 1.0813 |
| ensemble | changed | 8659 | 55.3297 | 85.4833 | 0.3354 | 1.3636 |

A current true label defines these subsets for diagnosis only. It is not an inference input. Endpoint equality does not rule out an intermediate change and return. Subsets differ across horizons. At +5 s, unchanged top-3 and NLL deteriorate slightly even though its top-1 improves.

## Per-pilot +5-second accuracy

| pilot_id | frequency | gru | transformer | ensemble | gain_points |
| --- | --- | --- | --- | --- | --- |
| pilot_001 | 8.4 | 45.24 | 47.77 | 54.06 | 6.29 |
| pilot_002 | 20.58 | 63.33 | 63.79 | 75.6 | 11.81 |
| pilot_003 | 2.12 | 50.97 | 57.96 | 63.9 | 5.94 |
| pilot_004 | 44.67 | 58.68 | 66.9 | 67.79 | 0.9 |
| pilot_005 | 3.64 | 56.64 | 61.3 | 74.16 | 12.86 |
| pilot_006 | 14.33 | 48.18 | 55.17 | 62.34 | 7.16 |
| pilot_007 | 0.77 | 51.28 | 53.65 | 54.42 | 0.77 |
| pilot_008 | 29.78 | 36.36 | 38.77 | 46.08 | 7.31 |
| pilot_009 | 3.14 | 57.46 | 59.03 | 69.03 | 10.0 |
| pilot_010 | 59.07 | 46.24 | 53.66 | 66.08 | 12.42 |

All 30 pilot/horizon top-1 comparisons improve. This does not demonstrate unseen-pilot generalisation or establish significance for every pilot individually. Complete per-pilot/horizon values are in data/summary_metrics.csv.

## Every catalogue entry at +5 seconds

| Label | Support | Old precision | New precision | Old recall | New recall | Old F1 | New F1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| turn_cold | 5437 | 0.6422 | 0.6984 | 0.6009 | 0.6601 | 0.6209 | 0.6787 |
| lag_pursuit | 4872 | 0.5466 | 0.6051 | 0.5913 | 0.6227 | 0.5681 | 0.6138 |
| recommit | 4857 | 0.5662 | 0.632 | 0.5958 | 0.6687 | 0.5807 | 0.6499 |
| extend | 2676 | 0.5236 | 0.6569 | 0.5254 | 0.5381 | 0.5245 | 0.5916 |
| pursue_target | 2561 | 0.489 | 0.5386 | 0.4592 | 0.5935 | 0.4736 | 0.5647 |
| beam_target_left | 1693 | 0.6795 | 0.7138 | 0.5747 | 0.71 | 0.6227 | 0.7119 |
| beam_target_right | 1445 | 0.5782 | 0.7416 | 0.5758 | 0.6893 | 0.577 | 0.7145 |
| secondary_shot | 1414 | 0.5316 | 0.619 | 0.6365 | 0.6768 | 0.5793 | 0.6466 |
| lead_pursuit | 1350 | 0.4426 | 0.5156 | 0.3941 | 0.5644 | 0.4169 | 0.5389 |
| crank_target_left | 1150 | 0.471 | 0.6125 | 0.6791 | 0.6887 | 0.5563 | 0.6484 |
| crank_target_right | 947 | 0.747 | 0.8172 | 0.5829 | 0.7645 | 0.6548 | 0.79 |
| short_range_attack | 216 | 0.4 | 0.4015 | 0.1204 | 0.2454 | 0.1851 | 0.3046 |
| descend_to_altitude | 162 | 0.4258 | 0.5284 | 0.5494 | 0.7469 | 0.4798 | 0.6189 |
| launch | 130 | 0.3398 | 0.5524 | 0.2692 | 0.4462 | 0.3004 | 0.4936 |
| decelerate_to_speed | 126 | 0.3059 | 0.5172 | 0.2063 | 0.2381 | 0.2464 | 0.3261 |
| notch_left | 124 | 0.3636 | 0.4884 | 0.2581 | 0.3387 | 0.3019 | 0.4 |
| abort_support | 113 | 0.3412 | 0.5612 | 0.2566 | 0.4867 | 0.2929 | 0.5213 |
| notch_right | 80 | 0.2857 | 0.7027 | 0.15 | 0.325 | 0.1967 | 0.4444 |
| maintain_heading | 36 | 0.425 | 0.525 | 0.4722 | 0.5833 | 0.4474 | 0.5526 |
| turn_to_heading | 20 | 0.5 | 0.4194 | 0.35 | 0.65 | 0.4118 | 0.5098 |
| beam_missile_right | 20 | 0.0 | 0.1224 | 0.0 | 0.6 | 0.0 | 0.2034 |
| support_missile | 19 | 0.0 | 0.6667 | 0.0 | 0.2105 | 0.0 | 0.32 |
| beam_missile_left | 6 | 0.0 | 0.5 | 0.0 | 0.1667 | 0.0 | 0.25 |
| search | 2 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| climb_to_altitude | 1 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| defensive_reversal | 1 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| drag_missile | 0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| accelerate_to_speed | 0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| commit | 0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| dive_defense | 0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| preemptive_crank | 0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| last_ditch_break | 0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| lock_target | 0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |

Seven entries have zero truth support. Small-support estimates are unstable, and zero-support performance is not measured. The slide class groups are chosen by support (at least 500 versus fewer than 500), not by improvement size. All 33 labels remain scored. Class metrics for all four methods and all three horizons are in data/class_metrics.csv.

## Probability quality

| model | horizon_s | count | top1_accuracy | top3_accuracy | nll | brier | ECE (points) | Mean confidence (%) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| frequency | 1 | 30802 | 0.1784 | 0.50032 | 2.39214 | 0.88743 | 0.49347 | 18.33322 |
| frequency | 2 | 30466 | 0.17994 | 0.5041 | 2.38989 | 0.88664 | 0.47782 | 18.47165 |
| frequency | 5 | 29458 | 0.18457 | 0.51483 | 2.37988 | 0.88405 | 0.42292 | 18.8797 |
| gru | 1 | 30802 | 0.54399 | 0.86946 | 1.26061 | 0.58563 | 6.88292 | 61.28198 |
| gru | 2 | 30466 | 0.54008 | 0.86723 | 1.27681 | 0.59121 | 6.98701 | 60.99475 |
| gru | 5 | 29458 | 0.51565 | 0.85036 | 1.33632 | 0.61233 | 6.74563 | 58.31057 |
| transformer | 1 | 30802 | 0.58967 | 0.884 | 1.22159 | 0.55519 | 8.75169 | 67.70041 |
| transformer | 2 | 30466 | 0.5838 | 0.88223 | 1.23615 | 0.55877 | 9.11054 | 67.49 |
| transformer | 5 | 29458 | 0.559 | 0.86513 | 1.29034 | 0.58298 | 8.73553 | 64.63545 |
| ensemble | 1 | 30802 | 0.65616 | 0.90105 | 1.11552 | 0.46943 | 2.93902 | 63.30893 |
| ensemble | 2 | 30466 | 0.65171 | 0.89818 | 1.13244 | 0.47381 | 2.92972 | 63.38134 |
| ensemble | 5 | 29458 | 0.6348 | 0.89246 | 1.16425 | 0.49162 | 3.41619 | 60.83309 |

The ensemble improves Brier, NLL and ECE relative to both original neural models. The frequency baseline has very low ECE despite poor accuracy: near-constant low confidence can match an aggregate hit rate without distinguishing examples. ECE must be read with accuracy and proper probability scores. Reliability-bin counts and the underlying 10-bin definition are published.

## Development search: evidence about what helped

Internal screen values must not be compared directly with outer-validation scorecards. They use smaller fitting partitions and different selection origins. The first screen includes unsuccessful geometry, regularization and long-history transformer candidates.

| name | fold | seed | best_epoch | nll | mean_accuracy | macro_f1 | accuracy_1s | accuracy_2s | accuracy_5s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| control | 0 | 42 | 17 | 1.3836 | 0.5187 | 0.2322 | 0.5262 | 0.5224 | 0.5076 |
| geometry_mlp | 0 | 42 | 7 | 1.3793 | 0.5161 | 0.2357 | 0.527 | 0.5211 | 0.5002 |
| geometry | 0 | 42 | 11 | 1.4227 | 0.5023 | 0.2251 | 0.5135 | 0.5084 | 0.4848 |
| history20_aux | 0 | 42 | 8 | 1.3951 | 0.502 | 0.2139 | 0.512 | 0.5054 | 0.4887 |
| history20_mlp | 0 | 42 | 11 | 1.3959 | 0.5398 | 0.2673 | 0.5495 | 0.545 | 0.5248 |
| history20_mlp_wide | 0 | 42 | 4 | 1.3925 | 0.512 | 0.2367 | 0.5209 | 0.5163 | 0.4987 |
| history20 | 0 | 42 | 8 | 1.3992 | 0.4983 | 0.2057 | 0.5088 | 0.5026 | 0.4835 |
| history20_wide | 0 | 42 | 5 | 1.4227 | 0.5032 | 0.2007 | 0.5104 | 0.5065 | 0.4928 |
| regularized | 0 | 42 | 16 | 1.4237 | 0.4971 | 0.2076 | 0.5077 | 0.5009 | 0.4827 |
| control | 1 | 42 | 12 | 1.3362 | 0.5138 | 0.2061 | 0.5267 | 0.5201 | 0.4946 |
| history20_mlp | 1 | 42 | 8 | 1.3368 | 0.5551 | 0.2399 | 0.5646 | 0.5605 | 0.5403 |

Fixed-mixture confirmation:

| name | fold | nll | mean_accuracy | macro_f1 | accuracy_1s | accuracy_2s | accuracy_5s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| control | 0 | 1.3836 | 0.5187 | 0.2322 | 0.5262 | 0.5224 | 0.5076 |
| long_pair | 0 | 1.3365 | 0.5905 | 0.3392 | 0.599 | 0.5942 | 0.5782 |
| accuracy_all | 0 | 1.3089 | 0.5909 | 0.3316 | 0.5992 | 0.5952 | 0.5783 |
| control | 1 | 1.3362 | 0.5138 | 0.2061 | 0.5267 | 0.5201 | 0.4946 |
| accuracy_all | 1 | 1.2072 | 0.6287 | 0.3253 | 0.6396 | 0.634 | 0.6126 |
| long_pair | 1 | 1.2249 | 0.6287 | 0.3241 | 0.639 | 0.6337 | 0.6135 |

The final four-variant mixture averaged 60.98% internal accuracy versus 60.96% for the wide/long pair. That small gap is not decisive evidence that twelve final models are necessary. Experiments support the combined procedure, not an isolated causal attribution to each feature.

## Uncertainty and evidence limits

| horizon_s | accuracy_gain | lower_95 | upper_95 | scenario_groups | bootstrap_draws |
| --- | --- | --- | --- | --- | --- |
| 1 | 0.06649 | 0.05599 | 0.07839 | 45 | 5000 |
| 2 | 0.06791 | 0.05692 | 0.08037 | 45 | 5000 |
| 5 | 0.0758 | 0.06386 | 0.08953 | 45 | 5000 |

The paired 95% bootstrap intervals use 5,000 resamples of 45 complete scenario groups. All pilots and overlapping windows in a group stay together. The original validation set informed the initial experiments, and the internal search was adaptive. These are development intervals, not a correction for all historical selection or a final test. The reserved test is untouched. No new model, temperature or mixture weight was fitted for this expanded analysis.

The original GRU training process exited with code 1 after writing results; independent fresh-process reloads subsequently reproduced all saved probabilities exactly. This documented runtime event is retained in the initial experiment guide. All experiments use idealised simulator tracks from the same ten known pilots; sensor noise and unseen policies remain untested.
