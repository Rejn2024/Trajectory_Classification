# Forecasting insights: speaker notes

## 01. Forecasting trained pilots from a tactical viewpoint

The dataset records the tactical aircraft B01 observing frozen trained policy A01. Forecasting is a separate supervised task: no RL retraining was needed for these experiments. Motion errors and classification accuracy have different units and weighting, so they cannot be ranked against each other.

## 02. B01 supplies the observations. A01 is the forecast target.

Both aircraft are simulated F-16s. The corpus uses idealised exact simulator tracks, without radar noise, occlusion or track loss. Future physical states and decisions supply training targets, never input features. True current labels are used only for evaluation subsets, or as targets in an unselected auxiliary candidate.

## 03. Keep complete scenarios together when splitting the data

The seed-1729 outer split is stratified by requested initial range. Initial models used validation for epoch selection, so this remains a development set. Skill validation has 30,802 / 30,466 / 29,458 valid origins across the horizons. Motion has 31,029 / 30,693 / 29,685; label availability differs. Missing future targets are masked, not forward-filled. The final skill experiment prepares no test rows.

## 04. Learned motion forecasts improve on kinematic baselines

Errors are 3D Euclidean error averaged within an episode, then equally across 450 validation episodes. At +5 s, transformer position error is 19.092 m versus 63.943 m for constant acceleration. GRU / transformer position errors at +1 s are 0.440 / 0.484 m. This was one seed per architecture; it does not establish a universal architecture ranking. The log scale makes all four baselines visible.

## 05. A skill forecast predicts a selected catalogue entry

Selections are joined at exact recorded times. There is no nearest-time join or carry-forward of a missing selection. Catalogue aliases such as lag_pursuit, turn_cold, extend, recommit and pursue_target can share guidance implementations. The label task measures selected policy entries; it is not a validated taxonomy of distinct physical manoeuvres or tactical quality.

## 06. Exact-label accuracy rises at all three forecast horizons

Each valid origin has equal weight. Ensemble top-1 scores are 65.62%, 65.17%, 63.48%; original transformer scores are 58.97%, 58.38%, 55.90%. The ensemble uses more history and computation. No labels were merged or dropped to increase accuracy. The train-frequency baseline predicts class frequencies learned only from training targets.

## 07. The harder changed-selection subset improves too

Changed-subset support is 2,264 / 4,254 / 8,659 at +1 / +2 / +5 s. These are different subsets; the rising line is not evidence that longer horizons are intrinsically easier. A selection can change and return between endpoints. The true current label is used to define this diagnostic only and is not an inference input. No future switch time is predicted by this experiment.

## 08. All ten pilots improve, with uneven gain sizes

These point estimates show the improvement is not confined to one pilot. They do not establish statistical significance for each cell individually, nor rank the pilots' tactical competence. Training and validation include the same ten trained policy identities under disjoint scenario groups.

## 09. The gain also appears in class balance and log-loss

Macro-F1 assigns equal weight to all 33 labels and zero to undefined F1; 26 labels have validation truth support. NLL assesses probabilities assigned to the correct label; lower is better, but it is not a complete calibration assessment. One counterexample to universal improvement is the +5 s unchanged subset: top-3 accuracy falls from 91.12% to 90.81%, and NLL rises from 1.0575 to 1.0813.

## 10. The selected improvement combines four model variants

Initial models use five seconds of observations; the new ensemble uses up to sixty. Full raw dependencies include an extra 0.5 s for backward acceleration. Each token uses only past or current observations, with episode-start masking. This is not an architecture-only comparison or a factorial ablation. The six-model wide/long alternative had almost identical internal accuracy (60.96% versus 60.98%); the selection rule chose twelve. The CPU timing is 30 warmed calls with four threads on one machine, including feature construction; it is not a deployment latency guarantee.

## 11. All new model selection stayed inside the training groups

The search was adaptive within training: an initial nine-candidate screen, an accuracy-focused follow-up, and a recorded fixed-mixture extension. Two internal partitions reduce reliance on one split but are not independent replications. Eligibility guards constrain deterioration in NLL and macro-F1. Every final normalization uses only the 210 training groups. The all-origin audit found zero future-frame or cross-episode reads. Reloads reproduce all saved probabilities exactly, and 76 preserved artifacts pass their hashes. These checks reduce specific leakage risks; they do not prove zero overfitting. The outer validation set was already used in the initial experiment.

## 12. Accuracy gains remain positive within the 95% intervals

Bootstrap draws resample complete scenario groups with replacement. The paired statistic compares new and old predictions on the same valid origins, retaining the population window weighting. Individual windows are correlated and are not treated as independent replicates. These intervals describe this development comparison and do not account for all historical adaptive model selection.

## 13. Useful forecasts; clear limits on what is established

These are offline simulation results for a known population under one scripted tactical observer. They do not validate real sensor deployment, tactical decision support, pilot competence ranking or unseen-policy generalisation. No new simulation, fitting, inference or test evaluation was performed to produce these slides. Suggested future work is not a completed experiment.
