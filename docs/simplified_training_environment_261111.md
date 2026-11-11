# Simplified training environment: 11 November 2026

Branch: `simplifed_training_environment_261111`, based on
`retraining_environment_261111` at `730d2d2`.

The current catalogue is **v4.0.0**: `change_trajectory` and `launch`.
Trajectory control combines signed heading, altitude and speed deltas into one
skill. Forecasting now targets future speeds, headings and altitudes through
continuous GRU and Transformer regression. See the
[current schema, training and forecasting guide](trajectory_state_forecasting.md)
for complete parameter, target, data-collection and checkpoint contracts.

Earlier branch revisions used four absolute-setpoint skills (v2), then four
relative-command skills (v3). Those pilot weights require retraining for v4.
The original 33-name catalogue contained sixteen zero-guidance skills and several
names with identical executed controls; none are runtime aliases in the new
catalogue. Historical experiments, datasets and reports keep their original labels.

Notebooks 04–06 train fresh pilots in
`artifacts/rl_pilot_simplified/{single,combat,hybrid}/<timestamp>/`.
Notebook 29 is the current continuous forecasting workflow. Notebook 04 rejects
existing training outputs instead of deleting them. Earlier catalogue and
classification notebooks require their original source versions to reproduce.

The earlier v2 full suite recorded 905 passes, 17 skips and seven historical
notebook failures. Those seven reproduced against unchanged source `730d2d2` and
concern saved notebook outputs, empty cells and outdated default assertions.
Current validation is reported in the branch handoff; old test totals are not a
claim that subsequent revisions ran the full suite.
