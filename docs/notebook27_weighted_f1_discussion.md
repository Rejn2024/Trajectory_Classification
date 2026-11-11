# Notebook 27: final reports and priorities for improving weighted F1

Conversation summary, 11 November 2026.

This note concerns
[`27_skill_intent_target_relative_missile_hgt_nn_duel_skill_forecasting.ipynb`](../notebooks/27_skill_intent_target_relative_missile_hgt_nn_duel_skill_forecasting.ipynb)
on branch `forecasting_ed_261104`. The results reviewed were committed at
`32fe999eef0f3a2464fa0fa45d4e4435a792033b`.

The initial discussion explained the final-report cell before saved outputs were
available in the inspected notebook. A subsequent refresh supplied a completed
run. The numerical findings below refer to that refreshed version. This review
did not run new training, generate flights, or establish a causal ranking of
architectures.

## 1. What the final-report cell displays

The cell begins with:

```python
required_finals = ["final_metrics.csv", "final_per_class.csv", "selection.json",
                  ...]
```

It evaluates the selected checkpoint, or reuses saved final predictions when
all required files exist and `REEVALUATE_SAVED=False`. It does not train another
network. Its eight displayed tables appear in this order:

| Table / saved file | Meaning |
| --- | --- |
| `final_metrics.csv` | Exact prediction among the original 33 skills, including left/right. |
| `final_intent_metrics.csv` | Auxiliary-head prediction of offensive, defensive or context-dependent intent. These are frozen catalogue groupings, not independent observations of latent tactical intent. |
| `final_family_metrics.csv` | Prediction among 29 families, collapsing four left/right skill pairs. |
| `final_hierarchical_credit_metrics.csv` | Family-first evaluation with partial credit for the correct family but wrong side. |
| `final_direction_metrics.csv` | Left/right prediction conditional on the **true** mirrored family; singletons and missing labels are excluded. |
| `training_frequency_baseline.csv` | Exact-skill baseline using retained training-label frequencies at each horizon. |
| `training_frequency_intent_baseline.csv` | The corresponding three-class intent baseline. |
| `intent_diagnostics/summary.csv` | Separate intent head versus intent probabilities obtained by grouping the skill head's probabilities. |

Ordinary summary tables have nine rows: train, test and validation, each at
forecast horizons +1, +2 and +5 seconds. Missing targets are masked separately
at each horizon. Direction reports use only eligible mirrored-family examples.
The final intent comparison normally has 18 rows because it compares two methods.

### Common columns

| Column | Interpretation |
| --- | --- |
| `split`, `horizon_s` | Evaluation partition and forecast lead time. |
| `count` | Number of valid labels, not number of independent flights. |
| `nll` | Mean negative log probability of the true class; lower is better. Confident errors receive a large penalty. |
| `accuracy` | Fraction of correct highest-probability predictions; higher is better. |
| `macro_f1` | Equal-weight average F1 over every configured class. Classes with no true support receive F1 zero. |
| `supported_macro_f1` | Equal-weight average F1 over classes with positive true support. |
| `weighted_f1` | Average class F1 weighted by true class support in this split/horizon. |

The final classification and credit reports are unweighted: training-time family
and handedness weights are not applied to their metric calculations. Unsupported
classes have undefined per-class NLL and F1 zero. Empty direction subsets have
count zero and undefined NLL/accuracy.

For mirrored skills, `P(skill) = P(family) * P(direction | family)`; singleton
skills use `P(family)`. Exact-skill reports take the largest joint skill
probability. Family reports take the largest family probability. These decisions
can differ because a mirrored family's probability is divided between two skills.
High conditional-direction accuracy is therefore not high end-to-end skill
accuracy.

### Hierarchical credit and intent diagnostics

With the configured equal family and handedness rewards, discrete family-first
credit is zero for a wrong family, 0.5 for a correct mirrored family with the
wrong side, and one for the correct family and side. Correct singleton families
receive one. Side correctness earns no credit without the correct family.

`family_accuracy` measures the family decision. `hierarchical_credit` averages
the discrete credit above. `expected_hierarchical_credit` averages probability
credit: for a mirrored target, `C = 0.5*p_f + 0.5*p_f*p_d`, where `p_f` is the
true family's probability and `p_d` the true side's conditional probability.
Singletons use `C = p_f`. `hierarchical_credit_loss` averages `-log(C)` per valid
example. It is distinct from exact-skill NLL and the balanced training objective.

Frequency baselines use add-one smoothing, `P(c) = (n_c + 1)/(N + K)`, separately
at each horizon. Every example gets the same distribution, so its hard prediction
is the most frequent training class. No held-out labels fit these probabilities.

The final intent comparison adds multiclass Brier score and ten-bin expected
calibration error (`ece`), both lower-is-better. Its `top3_accuracy` is one for
valid examples because the intent task has only three classes. Per-class CSVs
are also persisted but not displayed by this cell; they include support,
per-true-class NLL, precision, recall and F1.

## 2. Findings from the completed run

The notebook fits one network with seed 42. The selected checkpoint is epoch 37;
training stopped after epoch 47 because of NLL patience. Mean unweighted
exact-skill **test NLL** across horizons selects checkpoints. Thus the partition
named test is a development/selection partition. Validation is evaluated after
training, but its inspection in this discussion now makes it development evidence
for subsequent design choices.

### Exact-skill weighted F1

| Partition | +1 second | +2 seconds | +5 seconds | Equal-horizon mean |
| --- | ---: | ---: | ---: | ---: |
| Train | 0.4976 | 0.4976 | 0.4928 | 0.4960 |
| Test, used for selection | 0.4725 | 0.4720 | 0.4667 | 0.4704 |
| Validation | 0.4607 | 0.4603 | 0.4547 | 0.4586 |

Other observations:

- Validation family accuracy is 0.4661, 0.4651 and 0.4606. Conditional-direction
  accuracy is 0.9307, 0.9393 and 0.9197. Family recognition appears to offer more
  overall headroom than side recognition. These differently conditioned metrics
  do not provide an exact ceiling on weighted-F1 improvement.
- The train-validation weighted-F1 gap is about 3.7 percentage points. Training
  F1 itself remains around 0.50. More data could help, but these figures do not
  establish that sample scarcity is the dominant limitation; representation,
  objective, optimisation and incomplete observability remain plausible.
- Validation weighted F1 falls only 0.006 from +1 to +5 seconds. Longer-horizon
  prediction is not the only evident difficulty. This does not prove that the
  temporal encoder uses the supplied history effectively.
- The intent head's validation weighted F1 at +1 second is 0.6035, versus 0.5977
  from grouped skill probabilities. This supports its usefulness for the intent
  task, but does not isolate whether auxiliary supervision improves exact-skill
  F1; that requires an intent-loss ablation.

The source corpus contains 10,000 physical flights. Training retains approximately
137,000 windows from 4,000 flights/8,000 perspectives after non-overlap sampling.
Test and validation each contain 3,000 flights and over 1.2 million dense windows
at +1 second. Overlapping windows and the two views of a flight are correlated;
window totals should not be interpreted as independent sample sizes.

### Why the choice of F1 average matters

Weighted F1 is `sum_c((n_c/N) * F1_c)`. The five classes `recommit`, `turn_cold`,
`lag_pursuit`, `secondary_shot` and `extend` comprise about 81.5% of retained
+1-second training labels and 82.0% of +1-second test labels. Improving their
F1 therefore has much greater direct influence on this score than improving
extremely rare classes.

For illustration, raising each of those five class F1 values by 0.10, with all
other F1 values fixed, would increase test weighted F1 by approximately 0.082.
This is metric arithmetic, not a forecast of achievable improvement. Rare-class
data can also help indirectly by reducing false positives that hurt common-class
precision or recall.

## 3. First align the training and selection criteria

The actual run configuration uses family-balanced partial-credit loss, auxiliary
intent loss weight 0.25, family-weight ratio cap 100, and handedness-weight ratio
cap 20. The family audit contains rare-label weights around 59 versus about
0.59 for `recommit`. These are training choices, not evaluation weights.

Such balancing can support rare-class recall and family credit, but its benefit
to support-weighted exact-skill F1 is not established. Before attributing the
score to architecture, compare:

1. Current partial-credit loss against joint-skill cross-entropy.
2. Current family balancing against unweighted or less aggressively capped
   weighting, with other choices held fixed.
3. NLL-based selection against a predefined weighted-F1 selection rule, retaining
   NLL and calibration as quality checks.

The history already illustrates selection mismatch: epoch 40 has approximately
0.4731 mean test weighted F1 versus 0.4704 at the NLL-selected epoch 37, about
0.27 percentage points higher on the selection partition. This is not evidence
of a corresponding validation improvement, and the alternative checkpoint was
not evaluated on validation in this review.

## 4. Comparing the three improvement approaches

| Approach | Most useful opportunity | Main uncertainty | Suggested priority |
| --- | --- | --- | --- |
| Graph architecture | Preserve useful entity/geometry information and improve relational readout. | No evidence yet that graph expressiveness is the main bottleneck. | Focused ablations before a major redesign. |
| Temporal model | Improve history aggregation, local motion modelling or available context. | Small horizon degradation does not identify a temporal failure. | Small, controlled experiments early. |
| Additional flights | Add independent coverage of important situations and transitions. | Repetition from frozen pilots can leave the same gaps and ambiguities. | Preferred larger investment if learning curves support it. |

### a. Graph network

The current encoder already has two HGT layers, width 64, four graph heads,
typed aircraft/missile relationships, 18 physical edge features and a learned
threat prior. Adding depth or heads alone is not an evidence-based prescription.

Useful tests include a direct current-state feature bypass, alternative readout
of entity embeddings, retained entity tokens instead of early summary pooling,
and a threat-prior ablation. These distinguish information loss or an unsuitable
prior from insufficient model capacity. A temporal entity graph is a larger
change requiring reliable track association: current missile slots are not
guaranteed persistent identities.

Graph changes become more compelling if held-out errors concentrate in specific
geometries, multiple-missile situations or threat regimes. A graph cannot recover
unobserved opponent state simply by adding message-passing layers.

### b. Temporal model

The current input is 51 frames at 10 Hz, spanning about five seconds. A width-96,
two-layer, six-head Transformer processes the spatial frame representations;
the final contextualised position feeds the prediction heads.

Low-cost alternatives include attention pooling or horizon-specific readout, a
TCN, and a small GRU. Longer multiresolution input is a separate experiment:
retain dense recent observations and sparsely sample older history. It can help
infer engagement phase or persistent behaviour, but may reduce retained training
counts when non-overlap is enforced over the expanded history footprint.

Keep graph encoding, heads, loss, scenario partitions and forecast targets fixed
when first comparing temporal techniques. Compare training/inference cost and
parameter counts rather than assuming equal budgets. No temporal replacement can
recover details already discarded by frame-level spatial compression.

### c. Additional simulated flights

Prioritise new independent states and transitions relevant to common-family
errors: energy, aspect, range, missile exposure and pilot-pair coverage. Use
per-class precision/recall and confusion patterns on development groups to direct
collection. A campaign focused only on rare catalogue entries may improve
macro-F1 more than the weighted F1 requested here.

The existing source uses ten frozen pilots with deterministic inference. More
flights from them may still produce almost no positive examples of certain
skills. Adding new pilots or stochastic action sampling changes the prediction
distribution and can increase uncertainty as well as coverage; treat these as
separately versioned experiments. Different catalogue labels may also correspond
to similar executed controls, limiting what observable motion can distinguish.

Keep evaluation examples and their support weights fixed for the development
comparison. Use fresh reserved scenarios for the eventual independent assessment.

## 5. TCN: definition, advantages and limitations here

**TCN means Temporal Convolutional Network.** In the proposed experiment, it is
a stack of one-dimensional convolutions along time, applied to the sequence of
graph-encoded frame vectors. A common design uses causal, dilated convolutions,
residual connections, nonlinearities, normalisation and dropout.

- **Causal:** a representation at time `t` uses only observations at or before
  `t`. Left padding preserves sequence alignment.
- **Dilated:** a convolution samples positions separated by a dilation factor,
  allowing successive layers to combine evidence over increasing time spans.
- **Residual:** each block adds a transformed representation to a skip path,
  making deeper temporal processing easier to optimise.

The model would still predict future labels directly through the existing
+1/+2/+5-second heads. It would not receive future observations or labels.
Replacing the temporal encoder need not change the family, side or intent heads.
The present Transformer also sees only past input history; causal convolutions
are not a correction for demonstrated future leakage.

### Receptive field

The receptive field is the number of input positions that can affect one output.
For stride-one layers without temporal pooling, kernel size `k`, and `m`
convolutions per block sharing dilation `d_l`, it is:

```text
R = 1 + m * (k - 1) * sum_l(d_l)
```

For example, two convolutions per block, kernel size three, and dilations
`1, 2, 4, 8` give `R = 61`, covering all 51 current input frames when the last
valid output is used. This is an illustrative candidate, not a tuned recipe.
Larger or irregularly sampled histories need a suitable receptive field and
explicit time/gap information where relevant.

### Advantages

- Shared local filters provide a useful inductive bias for turns, acceleration,
  manoeuvre onset and other short temporal patterns.
- Dilations combine local evidence across the full window without a recurrent
  update at every time step.
- Convolutional training can process time positions in parallel; efficient
  streaming inference can cache the necessary past activations.
- A modest TCN may be easier to train or cheaper than a larger attention model,
  making it useful for both a competitive candidate and a diagnostic baseline.
- Its finite temporal footprint makes the context available to each output
  explicit and auditable.

### Limitations

- Information outside the receptive field is unavailable. Longer dependencies
  require more dilation/depth, longer input, or a separate memory mechanism.
- Connections between distant events pass through the convolutional stack;
  attention offers direct content-dependent interaction between supplied tokens.
- Large dilations can undersample fine event relationships unless local layers
  and dilation schedules are designed appropriately.
- Fixed filters assume temporal patterns can be reused across positions. Variable
  manoeuvre durations or irregular sampling may need multiscale branches or
  explicit timing features.
- Normalisation and padding must preserve causal alignment and handle any invalid
  history correctly. A streaming deployment must maintain caches across updates.
- Efficiency and data-efficiency advantages are hypotheses to measure. At only
  51 input tokens, attention's quadratic interaction cost may not dominate the
  existing graph computation, so end-to-end speed gains could be small.
- A TCN cannot repair missing labels, unobserved opponent decisions, spatial
  pooling losses or a training objective poorly matched to weighted F1.

There is therefore no basis yet to claim a TCN will outperform this Transformer.
It is a practical, bounded alternative whose benefits can be tested without
redesigning the complete model.

## 6. Recommended experimental sequence

1. Establish an objective/selection baseline aligned with exact-skill weighted
   F1, with several seeds. Predefine how horizons are aggregated; an equal mean
   of their weighted F1 values is one reasonable choice.
2. Measure a learning curve using, for example, 25%, 50% and 100% of training
   scenario groups with the same model. An upward trend supports more flights;
   a plateau motivates investigation of objective, representation and observability.
3. Compare the current Transformer, an improved readout and a modest TCN under
   the same spatial encoding and training protocol.
4. Add a targeted batch of training flights and measure its marginal improvement
   on unchanged development examples.
5. Pursue graph modifications if error analysis identifies a spatial limitation.

Report every horizon, common-class precision/recall, macro-F1, NLL and compute
cost alongside the primary weighted F1. Use whole-flight/scenario resampling for
paired uncertainty estimates; training seeds measure a different variability
source. Keep both perspectives of every flight in the same partition and reserve
fresh scenarios for a final independent evaluation.

## Implementation references

- [Notebook and saved outputs](../notebooks/27_skill_intent_target_relative_missile_hgt_nn_duel_skill_forecasting.ipynb).
- [Metrics, losses, checkpoint selection and final evaluation](../bvr_behavior_prediction/paired_skill_forecasting.py).
- [Graph encoders, temporal Transformer and hierarchical heads](../bvr_behavior_prediction/enemy_missile_models.py).
- [Intent taxonomy, skill hierarchy and intent comparison](../bvr_behavior_prediction/skill_intent_forecasting.py).
- [Non-overlap data preparation](../bvr_behavior_prediction/nn_duel_skill_forecasting.py)
  and [workflow documentation](nn_duel_skill_forecasting.md).
- [Recorded corpus configuration](../shared/corpora/nn-duels-20k-v1/eee3474c5493/config.json).
- Existing [TCN implementation](../bvr_behavior_prediction/models/tcn.py) and
  [earlier sequence experiments](skill_forecasting_sequence_experiments.md).
  These are starting references, not measured replacements for notebook 27.
