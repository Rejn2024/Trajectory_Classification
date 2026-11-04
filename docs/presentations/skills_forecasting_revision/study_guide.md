# Skills forecasting: slide-by-slide revision

Work through one numbered slide at a time. Explain the idea in your own words, then try its revision question. Answers are provided for checking after discussion.

## 01. What are we trying to forecast?

**Takeaway:** We predict a trained pilot's future catalogue choice from the tactical observer's available history.

Use this deck as a sequence of lessons. First establish who observes whom, how the forecasting corpus was collected, and which observations are available at prediction time. Then follow those observations through the MLP, TCN and Transformer architectures before reading training curves and evaluation metrics. The comparison focuses on four milestones rather than every search candidate. The current fifteen-model MLP/TCN ensemble has the highest measured accuracy among these four. The newer six-Transformer ensemble is smaller and has better probability quality, but did not improve validation accuracy. Neither result establishes an optimal architecture. Detailed experiment records remain available behind the presentation.

**Revision question:** What information goes into the forecaster, and what exactly comes out?

<details>
<summary>Check your answer</summary>

The input is the observation history available to tactical aircraft B01. The output is a probability distribution over 33 catalogue labels at each of +1, +2 and +5 seconds. These predictions describe the trained pilot's selected behaviour; they do not rate its competence.

</details>

## 02. Two different systems learn two different jobs

**Takeaway:** B01 observes A01 after the trained pilot has been frozen; forecasting does not update that pilot.

A01 is controlled by one of ten trained RL policies, while B01 is the scripted tactical opponent and observer. The policies were trained and selected before generating the forecasting encounters. Each recorded encounter provides observable tracks for the predictor and the trained policy's catalogue selections for supervised targets. The forecaster has no PPO critic and does not change the aircraft policy while learning. All ten pilots contribute to the pooled forecast models; there is not one forecasting network per pilot. Pilot identifiers are used to report evaluation subsets, not supplied as model inputs. Predictability can vary with behaviour and scenario difficulty without ranking which pilot flies or fights better.

**Revision question:** How does training the RL pilot differ from training its skill forecaster?

<details>
<summary>Check your answer</summary>

The RL pilot learns a policy that chooses actions to obtain reward in simulated encounters. The forecaster learns from recorded observations and future catalogue labels after that policy is frozen. RL episode return and forecasting loss therefore measure different learning problems.

</details>

## 03. Why 3,000 encounters after 112,500 training flights?

**Takeaway:** The forecasting corpus contains 300 shared starting scenarios, each flown by all ten frozen pilots.

The 112,500 figure describes one full population training stage and excludes its evaluation flights and other training stages. Policies change during that process. The forecasting corpus instead records a fixed set of ten policies, making their behaviour a stable prediction target. Its 300 shared scenario groups split into 210 training, 45 development-validation and 45 reserved-test groups, giving 2,100, 450 and 450 encounters respectively. All ten pilot versions of one starting scenario stay in the same partition. Prepared training and validation contain 146,690 and 30,802 forecast origins. Those counts are not independent flights: overlapping histories and repeated labels create substantial dependence within an encounter.

**Revision question:** Why are the RL training-flight count and the forecasting encounter count different?

<details>
<summary>Check your answer</summary>

One complete population training stage uses 10 pilots × 150 epochs × 75 flights = 112,500 training flights. The forecasting dataset is collected later from 300 new starting scenarios × 10 frozen pilots = 3,000 encounters. Its many forecast windows are correlated observations within those encounters.

</details>

## 04. History, forecast horizon and elapsed time

**Takeaway:** A longer backward history is legitimate when every input was available at the forecast origin.

Draw the forecast origin at time t, with input history extending left and the three target times extending right. The original Transformer sampled 51 observations across five seconds; improved models use sparse 20- or 60-second histories with 26 or 66 observations. Longer duration therefore does not simply mean more tokens. Forecasts still begin at six seconds of encounter time. Missing earlier history is masked, and windows cannot cross flight boundaries. No true current catalogue label, future track or pilot identifier is an input. Later encounters can be easier for several reasons, including a changing mix of surviving encounters, scenarios and labels; their scores cannot isolate the benefit of additional history.

**Revision question:** Does using 60 seconds of observation history introduce future information?

<details>
<summary>Check your answer</summary>

It does not do so merely by being longer: a forecast at time t can use observations at or before t. The horizon specifies the future target time, while elapsed time specifies how far the encounter has progressed. Leakage would arise if future observations or unavailable labels entered the inputs or model-selection process.

</details>

## 05. What observations does the forecaster actually use?

**Takeaway:** The 91 features describe observed state, relative motion, energy, missiles and past changes.

The original 54 features include B01's pose and velocity, target-relative position and velocity, range and bearing, a hostile-missile indicator, four B01 missile-track slots, energy proxies and acceleration estimated from the preceding half-second. The improved representation adds 37 current- or past-derived geometric quantities, including line-of-sight, aspect, closest-approach, energy and missile-presence information. These calculations reorganise available observations; they do not expose the trained policy's internal choice. Known time gaps and validity masks describe the sampling layout. The observations are idealised simulator tracks, so the results do not yet establish performance with radar noise, track loss or uncertain sensor measurements.

**Revision question:** Give three concrete forecasting features, and one item that is deliberately unavailable.

<details>
<summary>Check your answer</summary>

Examples include relative range, relative velocity, target bearing, aircraft energy estimates and backward-looking acceleration. The improved inputs also include derived geometry such as line-of-sight motion and closest-approach estimates. The trained pilot's true current or future catalogue label is unavailable to the predictor.

</details>

## 06. A 3D history can become an ordinary feature row

**Takeaway:** Flattening keeps each feature in a fixed time slot; it does not erase temporal ordering.

Start with a toy table containing speed and range at three observation times. Flatten it consistently into speed-oldest, range-oldest, speed-middle, range-middle, speed-current and range-current. A dense layer can then combine values from different times because every column has a stable meaning. In the real MLP, each timestamp contributes 91 normalised features and a validity flag. A batch of histories becomes a two-dimensional batch-by-flattened-feature matrix. Invalid earlier observations are masked so padding cannot impersonate a real measurement. The output returns to three dimensions: batch × three horizons × 33 class logits. Future labels belong in the target table and are never appended to these input rows.

**Revision question:** How does a temporal MLP turn a history tensor into a 2D training table?

<details>
<summary>Check your answer</summary>

Each forecast origin supplies a time-by-feature table, and a batch stacks many such tables. The MLP flattens each origin's table into one row while preserving the column positions for every time and feature. Including validity gives 26 × 92 = 2,392 columns for 20 seconds, or 66 × 92 = 6,072 for 60 seconds.

</details>

## 07. The temporal MLP and the twelve-model ensemble

**Takeaway:** The v3 ensemble combines four MLP recipes, each trained with three fixed random seeds.

The flattened history first enters a learned projection, followed by nonlinear processing and three residual blocks, then the classification output. A residual connection carries an earlier representation forward while a learned branch adds a correction. The four selected recipes comprise three 20-second variants and one 60-second variant; they vary width and regularisation as well as history. Random seeds alter initialisation and training order, giving twelve fitted members in total. Every member predicts every pilot and every horizon. Temperature calibration softens or sharpens member probabilities before uniform averaging. The resulting v3 ensemble has 28,977,060 parameters; its gains cannot be attributed solely to using an MLP because features, history and model averaging also changed.

**Revision question:** What makes this MLP temporal, and why are there twelve models?

<details>
<summary>Check your answer</summary>

It is temporal because its input contains ordered observations from a history window. Each member has three residual blocks and eight linear layers overall, producing all three forecast horizons. Four history/width/regularisation recipes are trained with seeds 42, 43 and 44, and their calibrated probabilities are averaged.

</details>

## 08. The TCN adds a different view of temporal patterns

**Takeaway:** V4 averages 50% of the twelve-MLP mean with 50% of the three-TCN mean.

A temporal convolutional network applies learned filters across the ordered history. Its causal convolutions do not read later positions than the position being processed. Six blocks use dilation factors 1, 2, 4, 8, 16 and 32, giving a 127-token receptive field that covers the complete 66-token input. Current, final and pooled history representations feed the forecasting head. The three TCN refits add a different representation to the retained MLP predictions. Their group receives half the mixture weight, rather than three fifteenths. V4 contains fifteen models and 31,730,031 parameters. It improved validation accuracy over v3, while its calibration error worsened, illustrating why the full scorecard matters.

**Revision question:** Is v4 a uniform average of fifteen model outputs?

<details>
<summary>Check your answer</summary>

No: half the final probability comes from the mean of twelve MLPs and half from the mean of three TCNs. Each MLP therefore has weight 1/24, while each TCN has weight 1/6. The three TCNs share a 60-second architecture and differ by training seed.

</details>

## 09. What the six new Transformers do

**Takeaway:** Attention combines observed history; the classification head directly predicts all three future horizons.

For 60-second history, each origin supplies 66 observations with 91 features. The temporal variant retains every contextual token position in a flattened readout; the patch variant combines older observations in groups of five and recent observations in pairs, producing 17 tokens including the separate current token. Patches respect the change in sampling rate. Relative-time information and masks describe the observed history. Bidirectional attention stays entirely within data available by the forecast origin. Each model's classification head outputs three sets of 33 logits directly, without feeding true future labels into a decoder. Standard feed-forward blocks remain inside these Transformers. The frozen six-member ensemble has 18,908,886 parameters and contains no separate MLP or TCN forecasting members.

**Revision question:** How do the temporal and patch Transformers differ?

<details>
<summary>Check your answer</summary>

The temporal model keeps 66 contextual history tokens in its readout. The patch model groups nearby observations into 17 tokens before attention, keeping the current observation separate. Both use three encoder-style blocks, width 192 and four attention heads, and are trained from scratch.

</details>

## 10. Four milestones explain the experiment history

**Takeaway:** The experiments changed information representation, training choices and averaging as well as architecture.

The first milestone is the original single Transformer, using 54 features and five seconds of densely sampled history. V3 combines twelve MLPs using 91 features and sparse 20- or 60-second histories. V4 retains those MLPs and adds three 60-second TCNs with a fixed probability mixture. V5 combines three temporal and three patch Transformers, all using 60-second histories and the same 91 observed features. The complete candidate searches are retained in the evidence but need not dominate this study deck. Internal selection scores and outer-validation scores are different measurements and must not be joined as one learning curve. None of these comparisons proves an architecture optimum.

**Revision question:** Why does the experiment history not isolate a single winning architectural ingredient?

<details>
<summary>Check your answer</summary>

The original-to-v3 change added features, longer history and model averaging alongside the MLP architecture. V4 added TCN predictions to the retained MLP ensemble. V5 tested a smaller all-Transformer alternative under the same forecasting task, so its outcome concerns that complete recipe.

</details>

## 11. Which data can influence each training decision?

**Takeaway:** Later recipes were selected within original training, frozen, then evaluated on development validation.

The main split holds all ten pilot versions of each scenario together: 210 training groups, 45 validation groups and 45 reserved-test groups. Later searches use two partitions inside training, each with 147 fit, 21 stop and 42 selection groups. Preprocessing is fitted only on fit groups. Epochs and temperatures come from stop groups, while recipe rankings use selection groups. Final architectures, mixture weights, seeds and budgets are frozen before the new validation comparison. The original Transformer used outer-validation NLL for checkpoint selection, and subsequent studies have repeatedly inspected that validation set. The internal partitions also overlap. These safeguards reduce specific leakage paths, but the reported results remain development evidence rather than an untouched final test.

**Revision question:** What are the different jobs of fit, stop, selection, validation and reserved-test groups?

<details>
<summary>Check your answer</summary>

Fit groups train weights and normalisation; stop groups choose checkpoints and calibration; selection groups compare recipes. After the recipe is frozen, final models train on all 210 original training groups and are measured on 45 validation groups. The reserved 45 test groups remain unevaluated, and the original models had already used outer validation for stopping.

</details>

## 12. Cross-entropy, NLL and the learning curves

**Takeaway:** Read the training objective, uncalibrated stop NLL and calibrated final NLL as distinct measurements.

A correct-label probability of 0.9 gives NLL 0.105; probabilities 0.5 and 0.1 give 0.693 and 2.303. These are loss values, not percentages. Use the recorded v5 patch-Transformer curves to observe training progress and the relationship between accuracy and probability quality. Falling training objective with rising stop NLL can indicate increasing overconfidence even while the top predicted label improves. Accuracy-selected checkpoints need not minimise NLL. Temperature is fitted on stop groups after checkpoint selection. The forecasting curves pool all ten pilots; per-pilot epoch losses were not logged. The archived RL return curves describe how the aircraft policies learned, and are not forecasting cross-entropy curves.

**Revision question:** What does NLL measure, and why can the plotted losses disagree?

<details>
<summary>Check your answer</summary>

For a hard true label, NLL is −ln of the probability assigned to that label: lower is better. Training uses label smoothing, dropout and changing weights; stopping NLL uses hard labels and fixed end-of-epoch weights. The shown stop curves are also measured before temperature calibration, so their absolute values are not interchangeable with final NLL.

</details>

## 13. Accuracy, precision, recall and F1

**Takeaway:** Accuracy rewards the right top label; macro-F1 gives every catalogue label an equal place in the average.

A true positive is a correct prediction of the chosen skill. A false positive predicts that skill when another label is true; a false negative misses it when it is true. Precision asks how often predictions of this skill are right, while recall asks how many true examples were recovered. F1 balances both through the harmonic mean. Top-one accuracy checks whether the highest-probability label is correct, and top-three checks whether truth appears among the three highest probabilities. Catalogue macro-F1 retains all 33 labels, including unsupported labels with zero F1 under the fixed metric. Support counts and per-label results therefore matter when interpreting the average, especially for rare skills.

**Revision question:** With TP=60, FP=20 and FN=40 for one skill, what are precision, recall and F1?

<details>
<summary>Check your answer</summary>

Precision is 60/(60+20)=75%, and recall is 60/(60+40)=60%. Their harmonic mean is F1=2×60/(2×60+20+40)=66.7%. The reported catalogue macro-F1 averages the separate F1 values over all 33 labels.

</details>

## 14. How the main models compare on accuracy

**Takeaway:** V4 leads the four-model validation comparison; v5 Transformers did not improve its measured accuracy.

Compare the four main systems only on the same saved validation populations and target masks. Their mean accuracies are approximately 57.75% for the original Transformer, 64.76% for v3, 66.19% for v4 and 65.88% for v5. The primary mean weights the three horizon accuracies equally; each horizon itself weights eligible forecast origins equally. At +5 seconds, v4 scores 64.86% and v5 scores 64.50%. The latest Transformer result must remain visible even though its internal gain did not carry through to validation. These scores measure exact catalogue-label prediction under known pilots and new scenarios. They do not prove that the architecture with the highest point estimate is globally optimal.

**Revision question:** What is the main accuracy finding from the Transformer experiment?

<details>
<summary>Check your answer</summary>

The six-Transformer ensemble achieved 65.88% mean validation accuracy, compared with 66.19% for v4. That is a decrease of 0.31 percentage points despite positive internal confirmation gains. The uncertainty analysis later in the deck establishes neither improvement nor equivalence.

</details>

## 15. Accuracy is only one part of the scorecard

**Takeaway:** The smaller Transformer ensemble improves probability scores while losing some top-one accuracy and macro-F1.

At +5 seconds, v4-to-v5 NLL improves from 1.1243 to 1.0980, Brier from 0.4763 to 0.4744 and calibration error from 4.36 to 3.82 percentage points. Top-three accuracy rises from 89.86% to 90.14%, while catalogue macro-F1 falls from 0.3767 to 0.3639. The mean scorecard shows the same important trade-off. NLL and Brier evaluate the probability distribution; calibration compares confidence with empirical outcomes. A low calibration error alone can also occur for an uninformative near-constant predictor, so it should not replace accuracy or proper probability scores. Six Transformers contain 18.91 million parameters versus v4's 31.73 million; parameter count is not a latency benchmark.

**Revision question:** How can the Transformer be useful even when its top-one accuracy is slightly lower?

<details>
<summary>Check your answer</summary>

It assigns probabilities more effectively by the recorded NLL, Brier and calibration measures, and its top-three accuracy improves. It also uses about 40% fewer parameters than v4. Those advantages accompany lower macro-F1 and do not establish an accuracy replacement or a measured latency improvement.

</details>

## 16. Which trained pilots are easier to forecast?

**Takeaway:** Population averages hide substantial pilot differences and genuine regressions.

Use the latest v4-versus-v5 comparison to make gains and losses equally visible. At +5 seconds, the pure Transformer ensemble gains about 5.23 points for pilot 007 and 2.05 for pilot 010, but loses about 3.34 for pilot 008 and 2.95 for pilot 006. Four of ten point estimates improve; pilot 005 is unchanged at that horizon. Across all horizons, pilots 001, 006, 008 and 009 regress under both the pure Transformer and the separately recorded blend. The largest pure-Transformer loss is pilot 008 at +2 seconds, about 4.01 points. Shared scenarios make these comparisons dependent, and pointwise pilot intervals do not correct for checking many pilot/horizon combinations.

**Revision question:** Does a higher forecast accuracy mean the underlying pilot is more competent?

<details>
<summary>Check your answer</summary>

No: forecast accuracy reflects the predictability of that pilot's selected labels under these scenarios. Label mixtures, persistence and encounter difficulty all contribute. The latest pure Transformers improve four pilot point estimates at +5 seconds, while several others regress.

</details>

## 17. Per-skill F1 needs its support count

**Takeaway:** A large change for a rare label can be driven by very few correlated forecast origins.

The table retains every catalogue label and compares v4 with the latest pure Transformer ensemble at +5 seconds. Common labels dominate origin-weighted accuracy, while macro-F1 gives rare labels equal weight in its class average. The support_missile F1 change from about 0.429 to 0.160 looks large, but that horizon has only 19 true examples. Some labels have one or two examples and others have none. A dash for absent truth support is not evidence that a label would be predicted perfectly or never matters. The archived complete experiment reports also record predicted counts, precision, recall, and coverage by pilot, encounter and scenario. Catalogue entries can share visible guidance, so these are exact selection-label scores rather than guaranteed distinct-manoeuvre scores.

**Revision question:** Why must a skill's F1 score be read beside its number of true examples?

<details>
<summary>Check your answer</summary>

A score based on a handful of examples is much less stable than one supported by many scenarios and encounters. No true examples means that recall for that label is not measured, even though the fixed catalogue macro-F1 retains a zero contribution. Support here counts eligible forecast origins, which are correlated within flights.

</details>

## 18. What the temporal statistics actually tell us

**Takeaway:** Later-origin accuracy and label persistence describe the data; they do not prove that longer history caused a gain.

Temporal analysis distinguishes future horizon from elapsed encounter time. Restricting all horizons to common origins helps compare the available starting observations, although future label distributions still differ. Full validation decision streams repeat the same label across 91.77% of adjacent one-second pairs. Completed label runs have median duration five seconds, but 450 episode-ending runs are censored, so those duration summaries are conditional. Later origin bins contain a smaller and different surviving set of encounters, fewer supported labels and fewer endpoint changes. Their higher accuracy cannot isolate the effect of extra history. Matching labels at the two endpoints also does not rule out an intermediate switch and return. These descriptive statistics do not change the original forecasting metric.

**Revision question:** Why is the observed 91.77% one-second repeat fraction not a deployable forecasting baseline?

<details>
<summary>Check your answer</summary>

It is a statistic of the true catalogue decisions, and the forecaster does not receive the true current label. Repeating that privileged label would therefore use unavailable information. It also differs from exact +1/+2/+5-second forecasting on the eligible evaluation origins.

</details>

## 19. How certain are the measured differences?

**Takeaway:** The latest Transformer intervals cross zero; neither improved accuracy nor equivalence is established.

The paired analysis resamples all ten pilot encounters within each of the 45 validation scenario groups together. This preserves important dependence that an independent-window calculation would ignore. V4 improved mean accuracy over v3 by 1.44 percentage points, with a conditional 95% interval of approximately +0.90 to +2.08. The pure v5 Transformers differ from v4 by −0.31 points, with interval −1.18 to +0.64. A separately frozen 21-model blend, containing 75% Transformer probability and 25% v4, gains only +0.07 points with interval −0.63 to +0.84. That blend is secondary evidence and does not demonstrate a pure-Transformer replacement. No equivalence margin was registered, and the reserved test remains untouched.

**Revision question:** What does it mean when the 95% interval for an accuracy difference crosses zero?

<details>
<summary>Check your answer</summary>

The resampling analysis is compatible with both a positive and a negative difference under its assumptions. It does not prove that the models are equally accurate. These intervals are conditional on the frozen models and reused development data, not adjusted for the entire model search.

</details>

## 20. How much experiment work has been done?

**Takeaway:** Training effort is recorded per study; summed fit time is not project wall time or inference latency.

For a consistent study inventory, v3 alone contains 23 fits: eleven internal fits and twelve final refits, totalling 24.305 recorded minutes. The earlier 36-fit inventory additionally included the initial and v2 work and must not be labelled as 36 new v3 fits. V4 completed 37 new fits with 57.089 summed logged minutes. V5 completed 29 new fits, comprising 23 internal fits and six final refits, with 135.304 summed minutes. Cached baseline fits were reused rather than retrained and must not be counted again as new work. Shared-GPU contention affects these durations; differing candidate counts, budgets and model sizes prevent treating the totals as a fair architecture throughput benchmark.

**Revision question:** Why should the summed fit minutes not be read as total elapsed project time?

<details>
<summary>Check your answer</summary>

Several fits ran concurrently on the same GPU, so adding their timers counts overlapping time and includes resource contention. The recorded boundaries also exclude activities such as loading, setup, simulation, reporting and some evaluation. The values describe recorded training work, not an isolated speed comparison between architectures.

</details>

## 21. What remains uncertain, and what could improve?

**Takeaway:** Further improvement requires a new frozen training-only experiment with the same primary comparison task.

The current evidence concerns ten known frozen pilots, idealised tracks and exact catalogue selections. It does not demonstrate unseen-pilot generalisation, sensor robustness or a universal architecture optimum. Some labels share guidance and some have very sparse support. Repeatedly reused internal partitions and development validation can also favour choices that do not transfer. The recorded curves motivate a separate comparison of stronger regularisation or probability-aware checkpoint selection inside training. Matched ablations could distinguish history, features, readout and ensemble effects. Any expanded dataset or alternate skill grouping would require its own explicit baseline and should not silently replace this task. These are proposals, not results already achieved, and the present unsuccessful accuracy challenge remains part of the record.

**Revision question:** What would make the next performance claim more convincing?

<details>
<summary>Check your answer</summary>

Register the candidate changes and success criteria, select them using grouped training partitions, and preserve the current baseline. Report accuracy, probability quality, per-pilot and per-skill effects together. An untouched final test can assess the chosen procedure after development choices are settled.

</details>

## 22. Revision recap: what can we now explain?

**Takeaway:** V4 leads measured accuracy; the smaller Transformer challenger improves probability quality, with important limits.

Use this slide as the starting point for oral revision rather than a claim that every detail must be memorised. Ask the reader to identify the learner, the target, the available inputs and the unit of data splitting. Then trace a history through one architecture and explain how ensemble probabilities are formed. Ask why cross-entropy, accuracy, macro-F1 and calibration can move differently, and why pilot or skill support changes the interpretation. Finally, distinguish the measured Transformer point estimate from equivalence, a proven improvement or an architecture optimum. The earlier results, complete comparisons and PDFs remain preserved. Return to slide 1 and work through each revision question in sequence, using the notes and linked evidence when a concept needs another example.

**Revision question:** Can you explain the complete experiment in your own words, from B01's history to the reported result?

<details>
<summary>Check your answer</summary>

B01's past observations become feature histories used to predict a frozen pilot's catalogue choice at three horizons. Grouped training and fixed evaluation compare the main MLP, TCN and Transformer systems on the same labels and origins. V4 leads measured accuracy; the smaller v5 Transformer ensemble improves probability quality without establishing an accuracy improvement.

</details>
