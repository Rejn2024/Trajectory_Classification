# v4 experiment notes

## Slide 1

The comparator is the current twelve-model v3 ensemble, never the older Transformer. All saved metric checks use exactly matched origins and horizon-specific target masks. The mean-accuracy interval uses the same 5,000 shared scenario-group draws across all three horizons, averages the three gains within each replicate, and then takes quantiles. It does not average interval endpoints. Intervals are conditional on fixed models and this development set, without selection adjustment. ECE worsened at all three horizons (2.94/2.93/3.42 → 4.16/3.63/4.36 pp at +1/+2/+5 s); NLL and Brier improved.

## Slide 2

The screen compares predefined weights 0.25, 0.5, 0.75 and 1.0. The two shortlisted component families receive three seeds each on both internal partitions. The equal pair is also a preregistered recipe. No architecture is added after seeing the screen.

## Slide 3

Portable pilot_comparison.csv includes all 30 pilot/horizon comparisons with support and paired intervals. metrics.csv also includes pilot-specific NLL, Brier, ECE and F1. Equal-pilot and equal-episode accuracy are exported separately from the primary pooled-origin estimand.

## Slide 4

F1 is the harmonic mean of precision and recall. It describes prediction of simulator-selected catalogue labels, not pilot competence or a unique physical maneuver. The primary macro-F1 always averages all 33 catalogue classes. Displayed dashes only prevent unsupported labels from looking like measured recall performance.

## Slide 5

All displayed curves use seed 42 separately in the two internal partitions; they are not three-seed averages. They come from internal stop groups. Training uses label-smoothed cross entropy, while NLL evaluates unsmoothed predicted probability of the true label. These values therefore need not match numerically. Some confirmation fits ran concurrently on the same GPU; individual timers include shared-resource contention. Their sum is not end-to-end elapsed time or an isolated-model throughput comparison. Execution notes are included when recorded. The portable evidence supports a PDF rebuild without model checkpoints or simulator data.
