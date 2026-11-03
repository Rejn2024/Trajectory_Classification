# Transformer experiment notes

## Slide 1

The primary all-Transformer challenger is reported even when worse. A positive secondary blend does not answer the primary question by itself: it still retains the incumbent MLP/TCN models. Reported outputs were frozen before one validation invocation. The labels, origins, masks and horizons are unchanged. Intervals use 5,000 shared scenario resamples, averaging the three horizon gains within each draw. These are not selection-adjusted intervals or an untouched final test.

## Slide 2

Sparse history uses older one-second observations and recent half-second observations. Patches never cross that sampling boundary and keep current state separate; all original input tokens are represented. Fixed time/gap metadata adds no new sensor information. Attention can connect all observed historical positions without seeing future observations. The explicitly named hybrid adds a parallel full-history residual MLP and cannot be presented as an all-Transformer result. Primary and secondary recipes have independently reported member counts and parameters.

## Slide 3

The preliminary screen compares single fitted candidates with the cached 15-member v4 fold-0 reference, whose mean accuracy is 60.5334%. This value comes from internal_baseline_metrics.json. The final primary is a multi-seed ensemble selected from single-architecture three-seed ensembles and their predefined pair. Each partition versus v4: NLL <= v4 +0.02; mean F1/33 >= v4 -0.01; each horizon accuracy >= v4 -0.003 (-0.30 pp). Secondary: mean accuracy gain >0 in both partitions. F1 here is the mean of the three all-33-label macro-F1 scores. Quality guards are reported even if no pure model passed them and the best pure challenger had to be retained as a negative experiment. The secondary never replaces the primary result. Exported screening and confirmation tables include all candidates and mixtures, including rejected results. All internal references use historical internal fits, never incumbent final weights trained on selection scenarios.

## Slide 4

Per-pilot differences must not be read as independent experiments: the same scenario groups are shared across pilots. Intervals resample complete shared scenario groups. Pilot prevalence and encounter difficulty differ, so aggregate improvement need not hold for every pilot. The complete CSV contains all horizons, counts and intervals for primary and optional secondary, without a post-validation choice between them.

## Slide 5

The displayed skill metrics are population-level +5-second F1. Full CSVs retain every pilot, horizon and class. Rare-skill fluctuations may be large even when only a handful of examples exist, so scenario/flight coverage accompanies raw origin counts. F1 combines precision and recall; its catalogue macro-average does not weight common labels more heavily and retains absent labels with zero F1 in the unchanged metric.

## Slide 6

Timers describe the recorded fit loop and stop-side operations, excluding loading and setup. Concurrent use of one GPU can affect those durations. Final refits have training objectives only because their epoch budgets were frozen internally. Stop accuracy, training cross entropy and stop NLL are different measurements, especially with label smoothing. Source snapshots, exact protocol bytes and complete evidence support independent review and a portable PDF rebuild without model checkpoints or raw observations.
