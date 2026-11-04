# Skills catalogue comparison: speaker notes

## 01. What changed in skill forecasting—and how much?

Headline changes compare the original transformer with the frozen improved ensemble on identical validation origins. The task predicts catalogue choices, not pilot competence. Motion forecasting is a separate task covered in the earlier overview. This deck adds descriptive metrics from saved probabilities without refitting or changing model selection.

## 02. The prediction target and evaluation population stay fixed

All scores weight valid forecast windows equally within each horizon. Overlapping windows are correlated. The reserved test is untouched. Same known pilots appear under disjoint scenario groups, so these are not unseen-pilot results. Idealised simulator tracks have no radar noise or track loss. Missing future selections are masked rather than carried forward. Some catalogue names share guidance.

## 03. The improved system has a larger information budget

Initial neural models share width 128, two layers, dropout 0.1, seed 42, AdamW and no class weighting. Both use 51 tokens spanning 5 s; acceleration needs 0.5 s extra raw history. The ensemble has three 20-second variants and a 60-second variant, each with three seeds. Its total parameter count is about 66 times the original transformer's; this is not a measured latency ratio. Initial selection used outer validation; the new search used only original training groups before a frozen outer comparison.

## 04. Read accuracy, class balance and confidence together

F1 is the harmonic mean of precision and recall. Undefined F1 is zero; seven absent labels remain in macro-F1 (33). Supported F1 changes the averaging set, not the vocabulary. NLL uses natural logs and a 1e-12 probability floor. Brier sums squared errors across 33 classes (range 0 to 2, no division by 33). ECE is weighted absolute confidence–accuracy difference in ten equal-width top-label confidence bins. ECE depends on bins and may be small for an uninformative predictor. See comparison.md for definitions.

## 05. +1 second: all four methods on the same origins

These values reproduce the original saved metrics on matched origins and masks. The frequency reference uses training-only per-horizon class priors. F1 (33) and F1 (26) have different denominators and should not be interchanged. The ensemble changes history and compute as well as model design. This is development validation; the original baselines used it to choose checkpoints.

## 06. +2 seconds: all four methods on the same origins

These values reproduce the original saved metrics on matched origins and masks. The frequency reference uses training-only per-horizon class priors. F1 (33) and F1 (26) have different denominators and should not be interchanged. The ensemble changes history and compute as well as model design. This is development validation; the original baselines used it to choose checkpoints.

## 07. +5 seconds: all four methods on the same origins

These values reproduce the original saved metrics on matched origins and masks. The frequency reference uses training-only per-horizon class priors. F1 (33) and F1 (26) have different denominators and should not be interchanged. The ensemble changes history and compute as well as model design. This is development validation; the original baselines used it to choose checkpoints.

## 08. Percentage-point gain is not relative error reduction

Relative reduction refers specifically to the top-1 error rate, not a 17.19-point gain in accuracy and not the fraction of all individual errors corrected. Corrections and new errors can coexist; the next slide reports those paired outcomes explicitly. All calculations use unrounded values.

## 09. The ensemble fixes errors and introduces new ones

Corrected means the original transformer was wrong and the ensemble right; a new error means the reverse. Both-wrong predictions may choose the same or different incorrect labels. These counts are correlated forecast windows, not independent encounters. The difference between corrections and regressions exactly reproduces the top-1 gain numerator at every horizon.

## 10. Changed selections gain most; unchanged cases also matter

Changed support is 2,264 / 4,254 / 8,659 and unchanged support is 28,538 / 26,212 / 20,799. Different horizons have different subsets, so rising changed accuracy does not mean longer horizons are easier. Endpoint equality does not rule out an intermediate change and return. Full top-1, top-3, F1 and NLL for each cohort and neural model are in comparison.md. Improvements are not universal.

## 11. Every pilot improves; absolute difficulty still varies

Pilot IDs identify evaluation subsets only. The same ten policies appear in training and validation, under disjoint scenarios. Absolute accuracy and gain vary: for example some already high-accuracy pilots gain little, while other pilots gain more than ten points. All three-horizon per-pilot tables are portable.

## 12. A small set of labels dominates the validation windows

The 500-occurrence cutoff is a display rule, chosen by support rather than score gains. All 33 labels remain in evaluation. The five most frequent labels include turn_cold, lag_pursuit, recommit, extend and pursue_target, which can share guidance implementations. Predictability of private catalogue choices must not be interpreted as a validated set of distinct manoeuvres.

## 13. Frequent labels: compare F1 with the original models

The improved ensemble raises +5 s F1 for every frequent class compared with the original transformer. This is a descriptive point-estimate comparison, not a significance claim for every label. Precision, recall and F1 for all 33 entries are tabulated in comparison.md; CSVs include all models and horizons.

## 14. Sparse labels: show the gains alongside the small counts

The smallest samples cannot establish stable accuracy. Beam_missile_left has six occurrences; support_missile nineteen. A larger F1 change there is not stronger evidence than a smaller change on thousands of examples. Precision and recall help expose trade-offs such as beam_missile_right's low precision despite higher recall. No labels were merged or removed for scoring.

## 15. Confusion matrices reveal the remaining label mixtures

The top eight labels are chosen by pooled truth support. All other true and predicted classes are aggregated into Other 25, retaining all counts in each denominator. Its diagonal includes confusions between different labels inside that group and is not exact-label recall. Individual displayed-class diagonals are recall. The full ungrouped count matrices are included for all methods and horizons.

## 16. The ensemble improves probability scores over both networks

At +5 s, transformer to ensemble Brier changes 0.5830 to 0.4916 and ECE changes 8.74 to 3.42 points. The frequency baseline has ECE around 0.42 points because its low constant confidence nearly matches its low hit rate, despite much worse accuracy, NLL and Brier. ECE depends on binning and finite support; it should never be used alone to rank these forecasters. No validation temperature was fitted here.

## 17. Confidence is closer to observed accuracy after improvement

Confidence is the maximum predicted class probability. Accuracy is averaged within each fixed bin, not across classes. Zero-count bins are omitted from the plot and retained with count zero in the CSV. Marker area is a monotone function of support. Sparse-bin points are descriptive, without independent bin confidence intervals. The final ensemble uses temperatures fixed from internal stop groups.

## 18. The development search also records unsuccessful changes

The first split fitted 147 groups, stopped on 21 and compared on 42. Two-split confirmation improved mean top-1 from 51.63% for control to 54.75% for the NLL-selected 20 s MLP. A targeted follow-up explored accuracy-based stopping, smoothing and 60-second history. Individual accuracy-selected models failed the NLL guard; fixed mixtures improved accuracy and NLL. All stages were internal and adaptive, not independent replications. Detailed original and mixture scores remain in comparison.md.

## 19. The final gain comes with more history and computation

Three 20-second variants use widths 256, 512 and 256; the 60-second variant uses 256. Final epochs are 46, 32, 40, 40. No outer validation was used for their epoch count, temperature, weights or seeds. Total checkpoints are approximately 116 MB. A local warmed CPU check measured median 20.3 ms, four threads, 30 calls; no comparable original-model latency measurement is available. The source audit does not establish zero overfitting, and near-equal internal alternatives should not be overinterpreted. The internal comparison used two versus four variants with one seed each. Comparing corresponding six- versus twelve-member final refits remains proposed, not an already measured result.

## 20. The gain remains positive within scenario-level intervals

Bootstrap intervals describe the paired population accuracy difference and retain window weighting. They do not correct for the entire historical model search. New Brier/ECE and class comparisons are descriptive analyses of fixed saved predictions; they are not a fresh held-out confirmation. No interval is claimed for each individual class, pilot or confidence bin.

## 21. A stronger forecaster, with specific remaining limitations

No new model fitting, model selection, simulation, inference or test evaluation was performed for this presentation. Additional diagnostics reuse fixed saved probabilities. Original experiments, training artifacts and earlier presentations remain preserved. Future work is proposed, not completed.

## 22. How each RL pilot learned: training and validation

These are the ten RL pilots' tactical-adaptation training histories, before they were frozen to generate forecasting encounters. All ten checkpoint hashes match the corpus configuration. Blue is raw mean training return each epoch; green shows only actual validation evaluations (epoch 1 and every fifth epoch), joined for readability. Stars mark the recorded selected checkpoints. Training uses stochastic actions on 75 fresh cases, validation uses evaluation actions on 50 fixed cases. Their gap is not a direct supervised generalisation gap. Each pilot has different reward weights; separate y scales preserve within-pilot trends but cannot rank pilot competence. No smoothing, new simulation or new selection was performed.

## 23. How the original skill forecasters learned

Original GRU and transformer logs pool all ten pilots and average the three forecast horizons. Selected epochs are 15 and 20 respectively, using validation NLL and the configured minimum improvement threshold. Therefore the selected epoch need not be the absolute numeric minimum. Training loss is measured during parameter updates with dropout; validation uses a fixed end-of-epoch model without dropout. The widening gap is consistent with overfitting, not a test for information leakage. These are recorded uncalibrated learning losses, not the ensemble's final probability scores. Per-pilot forecast losses were not logged each epoch; slide 11 shows final per-pilot validation accuracy only.

## 24. Improved MLPs: learning inside the training split

All four selected variants are shown on both internal partitions, using the saved seed-42 search runs. No favourable run is singled out. These stopping groups are within the original training partition, separate from the outer development validation. Stars mark recorded accuracy-selected epochs, not minima of the plotted NLL. Each training objective uses its variant's label smoothing (0.02, 0.03, 0.08 or 0.02) and training-mode dropout, whereas stopping NLL is ordinary unsmoothed evaluation NLL before temperature calibration. Falling training objective with rising stopping NLL can flag deterioration in probability quality even when accuracy is stable. These pooled curves neither prove absence of overfitting nor establish per-pilot learning behaviour. Exact stop accuracy is also included in the portable CSV.

## 25. Final ensemble refits: twelve training curves

All twelve final component runs are shown, with no newly trained model. Their epoch counts were fixed from internal development before refitting on all 210 training groups: 46, 32, 40 and 40. Refits deliberately had no outer-validation callback or early stopping. Training objectives include variant-specific label smoothing, so compare trajectories within a panel rather than treating heights across variants as a ranking. There is no single jointly trained ensemble loss curve: its members were fitted separately, calibrated with previously fixed temperatures and their probabilities averaged. Final per-pilot forecast evaluation exists, but per-pilot epoch histories do not. Recovering complete such histories would require a separate instrumented experiment, preserving the existing results.

## 26. Temporal MLP: a history window becomes one table row

N is the number of forecast origins, or the mini-batch size during training. The 3D array organises examples, timestamps and features; it does not mean three spatial dimensions. Flattening keeps all feature values and their time positions as distinct columns. In the implementation each timestamp also has a validity flag, so 26 × (91 + 1) = 2,392 and 66 × 92 = 6,072 columns. Older samples are one second apart; the last five seconds use half-second spacing including the present. Unavailable history at episode start is zero-padded after normalisation and identified by the flag. The toy row omits flags and normalisation for clarity. Each ordinary feed-forward residual MLP outputs 99 logits, reshaped into three sets of 33; softmax operates separately per horizon. Twelve members' calibrated probabilities are then averaged. Features at time t or earlier form X; future skill labels form three target columns y and are never appended to X. Scenario-group splitting keeps overlapping history windows and all pilots from a scenario together.

## 27. Two learning problems, two sets of networks

The 112,500 figure is training flights for one complete ten-pilot, 150-epoch stage, not the total project simulation count and not the size of the forecasting corpus. Selected frozen policies subsequently generated 3,000 encounters, using 300 shared starting scenarios. The forecasting split is 210/45/45 scenario groups, equivalent to 2,100/450/450 episodes. Only training and validation forecast targets are prepared. The 146,690 training origins and 30,802 validation origins are overlapping windows, not independent flights. The pilot actor's transformer and the forecasting model are different trained networks with different inputs, objectives and outputs. Skill forecasting estimates an exact symbolic policy choice; it does not directly assess tactical competence, manoeuvre quality or survival.

## 28. Inside one skill-forecasting network

The worked standard model uses 26 observations of 91 features plus one validity flag per timestamp, flattened into 2,392 columns. The input linear layer projects to width 256 and applies GELU. Three residual blocks each compute x + f(x), where f is LayerNorm, Linear 256 to 512, GELU, dropout, Linear 512 to 256, dropout. A final LayerNorm and Linear 256 to 99 generate logits reshaped into three categorical outputs over 33 labels. Thus the configuration field layers=3 means three residual blocks; the complete MLP contains eight linear layers, not merely three dense layers. It has 1,428,835 trainable parameters. For width 512, each block expands to 1,024 internally. Time positions are encoded by distinct input columns; there is no recurrent state, attention, or positional embedding in the selected MLPs. All history ends at the prediction origin. The loss uses smoothed cross-entropy on each available future label and averages the horizons. All four final variants have auxiliary_weight=0. Forecast training updates only the supervised predictor, not the RL pilot, simulator, skill catalogue, or tactical opponent. The earlier GRU and transformer are comparison forecasters; neither is included in the selected ensemble.

## 29. Four variants, twelve separately trained members

The exact frozen members are standard20, wide20, smooth20 and standard60, each with seeds 42, 43 and 44. Feature counts include 91 numeric features and one validity flag at each timestamp. Parameters per member are 1,428,835; 4,430,435; 1,428,835; and 2,370,915. Final epochs are 46, 32, 40 and 40. Each variant's epoch budget is the rounded median of its two internal best epochs, fixed before refitting on all 210 training groups. Each temperature is the median of the two stop-set-fitted values, not fitted on outer validation. A single positive temperature rescales all 99 logits in a member before softmax independently over each 33-label horizon; it changes confidence but not that member's class ordering. Temperatures are 1.465412, 1.401350, 1.169110 and 1.423757. Dropout/smoothing/weight-decay are respectively .20/.02/.03, .30/.03/.05, .30/.08/.05, and .20/.02/.03. Each member predicts all pilots and all horizons, not one pilot or horizon. Uniform probability averaging gives 75% of members to the 20-second variants and 25% to the 60-second variant. The internal wide/long pair was nearly tied with the four-variant mixture, so twelve members are not demonstrated to be the minimum necessary deployment size.

## 30. Training time: what the saved timers actually measure

All 36 unique completed run summaries are included once: two original fits; nine v2 screening fits plus two v2 second-split fits; six v3 first-split fits plus five second-split fits; twelve final refits. Their elapsed_seconds sum to 1,900.371772 seconds, or 31.672863 minutes. Stage sums are 89.737079, 352.325676, 533.309299 and 924.999717 seconds. Original GRU/transformer timers are 31.114268 and 58.622811 seconds. Final refit totals by variant are 237.733035, 166.544699, 228.977007 and 291.744976 seconds. The RTX4090/CUDA13 environment is explicitly recorded in v3/environment.json and the original run metadata; the v2 directory has no separate environment file. Improved timers start after feature construction, loading, model setup and optimizer creation. They cover the epoch loop, internal stopping predictions where applicable, checkpoint writes, and temperature fitting, but stop before select-group prediction. Original timers additionally include their final validation/report work. Summing these timers is an inventory of fit-stage elapsed time, not GPU kernel time or a measured end-to-end project duration. Pilot PPO training and the 3,000 corpus-generation flights are separate costs. Short training time does not establish low deployment latency or good generalisation.

## 31. Experiment evolution: why the recipe changed

The chronology begins with five-second, 54-feature original GRU and transformer forecasters, selected on outer development validation by NLL. The v2 search then used only the original training groups, divided into 147 fit, 21 stop and 42 select groups. Nine first-split candidates tested geometry, regularisation, longer history, a current-skill auxiliary target, and temporal MLPs. The 20-second MLP improved first-split mean accuracy from 51.87% to 53.98%; many other candidates deteriorated. The targeted v3 search tested accuracy-selected epochs, stop-only calibration, wider and more-smoothed MLPs, and 60-second history. Single models gained accuracy but worsened NLL, motivating the registered extension to uniform probability mixtures. The long-history single model shown here is the highest-accuracy individual on internal split 1; all candidate results remain available in the existing search tables. First-split four-variant mixture accuracy was 59.09% with NLL 1.309, versus control 51.87% and 1.384. Two shortlisted mixtures and control were confirmed on a second grouped partition. Mean accuracy across both partitions was 60.9817% for four variants and 60.9598% for the wide/long pair; four variants had lower mean NLL (1.2581 versus 1.2807), while the pair had slightly higher macro-F1. The chosen recipe, epoch budgets, temperatures and three seeds were frozen before fitting all training groups and comparing against the original models on outer validation. Internal mixture confirmation used one seed per variant; the final ensemble uses three. Internal and outer populations differ and must not form one uninterrupted performance trend. The adaptive search supports the combined procedure, not isolated causal credit to longer history, geometry, calibration or MLP architecture. No new model training, checkpoint selection or reserved-test evaluation was performed for this review.

## 32. Every pilot: forecast accuracy and uncertainty

Left: top-1 accuracy using each horizon's available labels, not a common-origin subset. Right: paired difference between ensemble and original-transformer correctness on the same +5-second origins. Each pilot's interval resamples its 45 scenario groups 5,000 times, keeping all overlapping windows within an encounter together. These are pointwise descriptive 95% percentile intervals, not simultaneous intervals across ten pilots or correction for development selection. Do not claim a statistically established improvement for every pilot simply because all point estimates are positive. Supports and all three horizon intervals are published in pilot_review.csv. These pilots are present in training; this is not a held-out-pilot generalisation assessment.

## 33. Pilots differ in skill mix and forecast confidence

Each point is one of the ten known RL pilots. Dominant-label fraction is computed from validation truth to describe the label mixture; it is not a fitted baseline or inference input. Different rewards, episode durations and policy behaviours may affect forecast difficulty, but this plot does not identify their causal contribution. The right diagonal denotes equality of aggregate confidence and accuracy, not perfect calibration at every confidence level. Below it means aggregate overconfidence; above means underconfidence. ECE uses ten fixed bins separately for each pilot, so pooled ECE can conceal opposing errors. Per-pilot bin counts and uncertainty would be needed before considering a dedicated calibration intervention. Pilot identity is not a model input.

## 34. Which skills can we forecast for each pilot?

Recall is the fraction of true occurrences of this skill that receive the correct label. It is not one-versus-rest accuracy, which would be dominated by true negatives for rare skills. A dash means zero truth support and does not mean measured failure or success. An asterisk marks fewer than 50 true forecast origins; 50 is a visibility flag, not an adequacy guarantee, because many origins belong to the same flight or label run. The eleven columns are selected solely by pooled truth support of at least 500, matching slide 13. All 33 labels for both models and all horizons are published, including predicted counts and precision. Equal supported-class macro-F1 across pilots averages different class sets; retain the fixed-33 result and supports as well.

## 35. Per-skill results need precision, recall and support

Precision divides correct skill predictions by all predictions of that skill. Recall divides correct skill predictions by all true occurrences. F1 is their harmonic mean. Origin support and encounter support answer different questions: many consecutive windows in one encounter can repeat the same label. skill_support_context.csv also supplies scenario-group and pilot counts. Frequent-label rows are a readable extract; the existing rare-label slide and complete tables retain all labels. For zero truth support, a stored zero F1 is a fixed-catalogue scoring convention rather than evidence that recall was measured. Never advertise high one-versus-rest accuracy as rare-skill recognition.

## 36. Forecast horizon: compare the same starting observations

Left uses the intersection of origins with all three future labels, so the forecast start times are identical across horizons. Ensemble accuracy is 65.6019%, 65.1436% and 63.4802%; original-transformer accuracy is 58.8703%, 58.2898% and 55.8999%. Targets still differ because the future time changes. Right uses all eligible origins at each horizon, whose counts are 30,802, 30,466 and 29,458. Endpoint change is 7.35%, 13.96% and 29.39%. The current true label is privileged evaluation information, never a feature. A repeat-current-true-label reference would be an oracle diagnostic, not a deployable baseline. Same endpoints can hide a switch and return in between. A longer history is more past input; a longer horizon is a later future target. Neither should be confused with elapsed time since the encounter began. Common-origin supported-class counts are 25/25/26, so fixed-33 macro-F1 is the consistent vocabulary comparison in the published table.

## 37. Later encounter accuracy is higher—but the cases change

Bins are fixed around when a full 20-second and 60-second history becomes available: 21 and 61 seconds, accounting for acceleration lookback, reset exclusion and one-second forecast origins. Counts are 6,411, 11,531 and 11,516 origins. The early bin covers all 450 encounters and 45 scenario groups; the middle 389/39, and late only 225/24. These are selected surviving subsets, not repeated measurements of an unchanged evaluation population. Endpoint-change rates decline 40.01% to 32.70% to 20.17%, and supported labels decline 24 to 23 to 17. Accuracy rises, but late top-3 decreases from 90.48% to 90.06% relative to the original transformer. The complete table retains all horizons, NLL and both F1 definitions. A controlled history ablation on the same origins, models and training protocol is needed to attribute any effect to history length.

## 38. Skill selections persist, then switch

This analysis reads full decision streams for validation episodes only, skipping every training/test row group before decoding. There are 37,731 adjacent one-second pairs, 91.7733% with the same label, 3,104 switches and 3,554 runs. A run means consecutive identical catalogue selections. The 3,104 fully observed completed runs have median 5 seconds and 90th percentile 16 seconds; the 450 final runs are right-censored and are excluded from those duration quantiles. These are conditional summaries, not unbiased estimates of all label durations. Gaps would break and censor runs; no gaps were found in these streams. Full streams include early/terminal choices outside eligible forecast targets and contain 27 labels versus 26 in the forecast scorecards. The repeat fraction is a truth statistic, not a model's forecasting accuracy. Current labels remain unavailable at inference. Transition counts reveal common changes but do not establish that every label change is a distinct physical manoeuvre.

## 39. Exact catalogue labels can share visible guidance

The named cases share emitted guidance components in the inspected implementation; they are not a claim that entire trajectories, parameters, lifecycle semantics or tactical consequences are identical. The original corpus records exact selections, which remain the supervised labels. No entries were merged or rescored for this review. Such overlap helps explain why tactical catalogue names must not automatically be interpreted as separable manoeuvres from B01 observations. Establishing an observability ceiling would require a separate study. A supplementary behaviour-family or motion-consistency metric could be informative only after a semantic mapping is justified independently of which mapping raises validation performance.

## 40. A reporting structure that makes the claim reviewable

The established primary metric weights valid forecast origins equally. Equal-encounter accuracy first averages within an eligible encounter, then across encounters; equal-pilot accuracy first averages all eligible origins within a pilot, then across pilots. These need not agree because encounter lengths and label mixtures differ. The companion table reports all horizons and the original transformer as well. Use scenario-group resampling rather than independent-window standard errors. Per-pilot intervals are pointwise, and selection bias is not removed by a bootstrap. Validation remains development evidence; the reserved test should be assessed only after the next protocol is frozen. Method references: Cawley & Talbot 2010, Roberts et al. 2017, Field & Welsh 2007 and Guo et al. 2017; links and their application are in study_guide.md.

## 41. Next experiments: isolate causes before adding complexity

This is a ranked experimental agenda, not a claim that any proposed change has improved the saved results. Fix success criteria and compute budgets before each comparison, keep all ten versions of a scenario together, fit preprocessing only on the fitting partition, and retain failed runs. Data-size learning curves should use nested fitting-group subsets with common stop/select groups and multiple seeds. Matched ablations are required because the original-to-improved experiment changed architecture, features, history and ensemble size together. The measured internal comparison used two versus four variants with one seed each; a six-versus-twelve-member final comparison remains proposed. Rare-skill interventions may harm calibration or common-class accuracy, so natural-prevalence evaluation and all metrics remain necessary. Any future temporal-segment F1 must have separately defined alignment, matching and censoring; it is different from the per-label F1 already reported. No new training, calibration, model selection or reserved-test evaluation was performed for this review.
