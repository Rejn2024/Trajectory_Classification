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

Three 20-second variants use widths 256, 512 and 256; the 60-second variant uses 256. Final epochs are 46, 32, 40, 40. No outer validation was used for their epoch count, temperature, weights or seeds. Total checkpoints are approximately 116 MB. A local warmed CPU check measured median 20.3 ms, four threads, 30 calls; no comparable original-model latency measurement is available. The source audit does not establish zero overfitting, and near-equal internal alternatives should not be overinterpreted.

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
