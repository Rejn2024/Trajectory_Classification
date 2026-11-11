# Simplified training environment: 11 November 2026

Branch: `simplifed_training_environment_261111` (spelling preserved as requested).
Based on `retraining_environment_261111` at `730d2d2`.

## Catalogue v2

New pilots choose among four commands. The ordered skill head is
`turn_to_heading`, `set_altitude`, `set_speed`, `launch`.

| Skill | Learned parameter | Bounds | Executed request |
| --- | --- | --- | --- |
| `turn_to_heading` | `heading_deg` | 0–360° | Shortest signed heading error, converted to radians |
| `set_altitude` | `altitude_m` | 0–20,000 m | Requested altitude minus current altitude |
| `set_speed` | `speed_mps` | 30–800 m/s | Requested speed minus current speed |
| `launch` | None | — | Set the simulator's existing `shoot` request to 1 |

Altitude and speed commands support both directions. A setpoint equal to the
current value intentionally requests zero change; these commands remain useful
because other setpoints change the flight. The simulator still applies its existing
normalization, saturation, flight controller, ammunition and launch constraints.
A launch selection requests firing on each backend substep as before; it does not
guarantee a particular number of missiles. Requested speed control stays enabled
by the source branch's default configuration.

## Reduction from the 33-name catalogue

The reduction follows `TacticalSkill.execute()` and the RL backend's consumed
heading, altitude, speed and shoot outputs, rather than the old skill descriptions.

| Previous skills | New command |
| --- | --- |
| `maintain_heading`, `turn_to_heading` | `turn_to_heading` |
| `climb_to_altitude`, `descend_to_altitude` | `set_altitude` |
| `accelerate_to_speed`, `decelerate_to_speed` | `set_speed` |
| `launch`, `secondary_shot`, `short_range_attack` | `launch` |
| `beam_target_left/right`, `crank_target_left/right`, `notch_left/right`, `beam_missile_left/right` | Removed; select a desired heading through `turn_to_heading` |

`break_left` and `break_right` are also excluded. There are no runtime aliases for
removed names, so they cannot reappear as redundant learned actions.

These sixteen skills emitted zero heading, altitude, speed and shoot requests and
are removed: `pursue_target`, `lead_pursuit`, `lag_pursuit`, `turn_cold`, `extend`,
`recommit`, `search`, `lock_target`, `commit`, `support_missile`, `abort_support`,
`preemptive_crank`, `drag_missile`, `dive_defense`, `last_ditch_break`, and
`defensive_reversal`.

Only physically consumed setpoints remain in parameter schemas. Target IDs,
ranges, offsets, load factors, timeout and tolerance parameters no longer occupy
the NN's continuous action space. The backend plans on its existing fixed cadence
and does not use the skill completion flag to schedule decisions.

## How the neural network chooses setpoints

The existing transformer encodes the observation history. Its categorical head
has four outputs; the continuous head predicts Gaussian means for the three
effective numeric controls, with learned standard deviations. Training samples
raw values; deterministic evaluation uses their means. Each raw value `z` maps
to its public bounds using `low + (tanh(z) + 1) / 2 * (high - low)`.

The default shared continuous head predicts all three values from the same
observation representation. Decoding uses only the selected skill's parameter;
`launch` decodes to `{}`. The optional skill-conditioned parameter head remains
supported and also derives its dimensions from the installed catalogue.

Fresh runs default to `selected_skill_parameters=True`. PPO log-probabilities
include only the chosen skill's effective continuous controls, and entropy uses
the categorical expectation over those controls. A launch sample has no Gaussian
contribution to its action log-probability. Existing reward definitions, opponent
policies and flight-control implementations are retained.

## Train fresh pilots

Install from the repository root, then restart the notebook kernel:

```bash
python -m pip install -e '.[ml,video]'
```

Run notebook 04 for a single pilot, 05 for the combat population, or 06 for the
hybrid population. All three print the runtime catalogue and explicitly enable
selected-skill parameter scoring. New output roots are
`artifacts/rl_pilot_simplified/{single,combat,hybrid}/<timestamp>/`, with separate
`jsbsim-simplified-*` MLflow experiments. `BVR_PILOT_OUTPUT` can override the path;
choose a fresh directory. Notebook 04 now rejects earlier training files rather
than deleting them. Editable training notebooks have cleared historical outputs;
preserved results remain in `notebooks/results/` and published artifacts.

For a short notebook-05 installation check, set `BVR_PILOT_COUNT=1`,
`BVR_PILOT_EPOCHS=1`, `BVR_PILOT_EPISODE_SECONDS=1`,
`BVR_PILOT_UPDATE_EPOCHS=1`, `BVR_PILOT_SCENARIOS=3`,
`BVR_PILOT_EVALUATION_SCENARIOS=3`, `BVR_PILOT_TEST_SCENARIOS=3`, and
`BVR_PILOT_WORKERS=1` before executing setup. Production defaults remain editable
in each notebook.

Each training `config.json` records the full versioned, ordered catalogue and
ordered numeric parameter names. Inference, checkpoint initialization and completed
pilot reuse require an exact contract match. Old 33-skill checkpoints and metadata
without the new contract are rejected with a retraining instruction. Existing
weights are not relabelled or converted. Historical corpora, reports and frozen
source snapshots keep their original labels; generate new data from the newly
trained pilots before fitting forecasts for this catalogue. Historical generators
such as notebook 02 use retired names and must be reproduced with their original
source version. Notebooks 04–06 are the training entry points for new pilots here.

## Validation

Regression tests cover catalogue removal, signed setpoints and normalization,
dynamic NN dimensions, selected-control probabilities and gradients, PPO updates,
checkpoint round trips and incompatible artifact rejection, and the three training
notebook workflows.

Validation used Python 3.12 and CPU PyTorch 2.15.0:

- Full `python -m pytest -q`: **905 passed, 17 skipped, 7 failed**. All seven
  failures were reproduced against the unchanged source commit `730d2d2` in an
  isolated source tree. They concern saved outputs or empty cells in historical
  forecasting notebooks 20, 21, 25 and 27, and two outdated default-setting
  assertions for notebook 03. They are unrelated to the new catalogue. The
  pilot-training, checkpoint and training-notebook checks pass.
- A real Python BVR Sim/JSBSim smoke run trained one pilot for one epoch on three
  one-second training flights, evaluated three validation cases, saved the
  checkpoint and loaded it through `load_trained_pilot()`. The restored policy
  has four skill outputs and three numeric control parameters, with catalogue
  version `2.0.0` in its saved configuration.
- Changed Python sources parse, notebooks 04–06 compile with cleared outputs,
  and `git diff --check` passes. CUDA-specific checks were skipped because this
  validation environment has no GPU.

This smoke run does not constitute a full population training experiment or
establish model-performance improvements.
