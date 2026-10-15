# Trained pilot episode dataset from the tactical observer

Raw encounter data for forecasting the behaviour of ten trained pilots from a
scripted tactical aircraft's perspective. Each frozen tactical-adapted pilot flies
300 encounters, giving 3,000 episodes. Both aircraft are F-16s.

The **observer is B01**, controlled by the scripted tactical policy. The
**prediction target is A01**, controlled by the trained pilot identified by
`pilot_id`. Native simulator colours are Blue for B01 and Red for A01; the RL
adapter uses the opposite colour names. Aircraft IDs define the recording roles.

This reverses the recording perspective of
[tactical-episodes-v1](../tactical-episodes-v1/README.md). The existing dataset is
retained separately. The same checkpoints and 300 starting scenario seeds are
used, while both controllers and the simulation remain unchanged. Encounters are
rerun to capture B01's own missile information and relative observations; the
observations are not produced by merely renaming the previous dataset's columns.

## Contents

| File or directory | Contents |
| --- | --- |
| `observations/pilot_NNN.parquet` | B01's 51-feature observation stream at 10 Hz, plus episode ID, step and time |
| `tracks/pilot_NNN.parquet` | Observer B01 and target A01 positions, velocities, alive flags and cumulative launch counts in physical units |
| `target_decisions/pilot_NNN.parquet` | Privileged A01 action records: selected skill and continuous parameters, normally at 1 Hz |
| `episodes.parquet` | Episode ID, target pilot, scenario group, seed, checkpoint hash, duration, initial state and final outcomes |
| `config.json` | Requested scenarios, recording roles, checkpoint hashes and frozen source provenance |
| `schema.json` | Column keys, coordinate system and role definitions |
| `metrics.json` | Completed counts, target outcome frequencies and verification checks |
| `manifest.json` | File sizes, SHA-256 checksums and software versions |
| `reproduction.zip` | Collection scripts, frozen simulator/policy source, assets and ten selected checkpoints |

Files use Zstandard-compressed Parquet, with one shard per pilot and one row group
per episode. `episode_id`, `step` and `time_s` align observation and track rows.
Time zero records the reset state; subsequent samples follow each 0.1-second
simulator step, including the terminal sample. Each encounter lasts at most 120
simulated seconds and may end earlier.

No future targets, manoeuvre labels, prediction horizons, overlapping history
windows, train/validation/test assignments or fitted normalisation are generated.
These preprocessing choices remain open. The actual target skill records are
logged actions, not precomputed forecasting labels.

## Observation definition

The recorder calls the simulator's existing compact observation extractor for
**B01**, then appends observer-first energy features using the same scales as the
learned-pilot adapter. The tactical controller itself accesses aircraft objects
directly; this vector is a recording interface, not its literal controller input.

| Columns | Meaning | Existing fixed scale |
| --- | --- | --- |
| `obs_00`-`obs_02` | B01 position north, west, up | Horizontal / 50,000 m; vertical / 10,000 m |
| `obs_03`-`obs_05` | B01 velocity north, west, up | / 600 m/s |
| `obs_06` | B01 altitude | / 10,000 m |
| `obs_07`-`obs_08` | B01 heading sine and cosine | Unitless |
| `obs_09`-`obs_11` | A01 position relative to B01 | Horizontal / 50,000 m; vertical / 10,000 m |
| `obs_12`-`obs_14` | A01 velocity relative to B01 | / 600 m/s |
| `obs_15` | Range from B01 to A01 | / 50,000 m |
| `obs_16`-`obs_17` | Bearing of A01 relative to B01 heading, sine and cosine | Unitless |
| `obs_18` | Live hostile missile targeting B01 | 0 or 1 |
| `obs_19`-`obs_46` | Up to four live B01-team missiles: relative position, relative velocity and range | Position scales as above; velocity / 1,200 m/s; range / 50,000 m |
| `obs_47` | B01 kinetic energy proxy | (speed / 400 m/s) squared |
| `obs_48` | B01 potential energy proxy | altitude / 12,000 m |
| `obs_49` | A01 kinetic energy proxy | (speed / 400 m/s) squared |
| `obs_50` | A01 potential energy proxy | altitude / 12,000 m |

Relative position and velocity use world North/West/Up axes, not a rotated body
frame. The bearing features are heading-relative. Empty missile slots are zero;
slot indices are not stable missile identities. The target block is zero when A01
is dead, and the compact 47-feature block is zero when B01 is dead. The appended
energy values still follow aircraft state queries. Alive flags and reset-state
initialisation transients are retained for later sample selection.

The interface provides idealised exact tracks without radar noise, occlusion or
track loss. It does not establish that all these quantities are observable by a
real aircraft. Physical trajectories are retained separately in `tracks/`.

## Forecasting use and episode outcomes

The intended forecasting input is B01's observation history up to prediction time.
The prediction subject is A01's future motion or behaviour. History length and
future horizons can be chosen later, including 1, 2 and 5 seconds.

`target_decisions/` contains A01's internal skill choices and parameters. These
are **not available to B01** and must not be included as observer input features.
They may support later analysis or target construction. Future track samples,
final outcomes, checkpoint identity and target pilot identity are also separate
from the observation vector. Supplying target identity would constitute a
different, identity-conditioned forecasting task.

Outcome fields are explicit about roles: `target_survived`,
`target_eliminated_observer`, `observer_survived`, `observer_eliminated_target` and
`observer_destroyed`. `target_outcome` and the outcome counts in `metrics.json`
are reported from the **learned target pilot's** perspective. A clean target win
means A01 survived and eliminated B01 with a successful missile. These outcomes
are metadata, not forecasting inputs.

The same 300 scenario seeds, 3,000,000-3,000,299, are used across all ten target
pilots. There are 100 requested starting ranges at each of 18, 30 and 45 km. This
is 3,000 encounters from 300 matched starting scenarios, not 3,000 independent
initial geometries. All outcomes are retained without selection for victories.

The native spawn manager samples the engagement axis and aircraft altitudes, with
approximately opposing courses and a fixed initial speed. The adapter applies
separation and an altitude interval; nominal scenario bearings, headings and
speeds are not directly applied. `initial_state_json` records actual conditions.

When creating evaluation splits later, keep all encounters sharing
`scenario_group` together and construct history windows within episodes. The
earlier opposite-perspective dataset contains matching encounters: if both
datasets are used, group them together as well. A held-out-pilot assessment is a
separate test of generalisation to unseen learned policies.

## Download and read

The data files and reproduction archive use Git LFS; small JSON metadata and
Markdown remain readable in ordinary Git. Compressed files are sharded and checked
against a conservative 1 GB limit per file. The manifest records actual sizes.

```powershell
git lfs pull --include="shared/corpora/trained-pilot-episodes-v1/**" --exclude=""
```

```python
from pathlib import Path
import pandas as pd

root = Path("shared/corpora/trained-pilot-episodes-v1")
filters = [("episode_id", "==", "pilot_001_case_0000")]
observations = pd.read_parquet(root / "observations/pilot_001.parquet", filters=filters)
tracks = pd.read_parquet(root / "tracks/pilot_001.parquet", filters=filters)
episodes = pd.read_parquet(root / "episodes.parquet")
```

## Collection and verification

The collector uses deterministic inference from the completed tactical runs'
validation-selected `best_model.pt` checkpoints. There are no PPO updates.
The learned controller's actual history is checked during collection, while the
published observation stream is recorded separately from B01. Saved observer and
relative target geometry are checked independently against physical tracks.
Checksums, finite values, sample alignment and matched initial states are verified
before packaging. Reproduction may vary slightly across hardware.

```powershell
.venv\Scripts\python.exe -X utf8 scripts/generate_tactical_corpus.py collect --perspective tactical --workers 8
.venv\Scripts\python.exe -X utf8 scripts/generate_tactical_corpus.py status --perspective tactical
.venv\Scripts\python.exe -X utf8 scripts/generate_tactical_corpus.py package --perspective tactical
.venv\Scripts\python.exe -X utf8 scripts/generate_tactical_corpus.py verify --perspective tactical
```

The default working directory is `artifacts/trained_pilot_episode_data_v1`, with
live counts in `progress.json`. Collection can resume with the same configuration.
Completed publication packages are not overwritten.

For reproduction from the archive, extract it into a fresh directory and run its
included `scripts/generate_tactical_corpus.py` with `--perspective tactical`,
`--job` pointing to the extracted directory, and fresh `--work` and `--output`
paths. Use the repository environment with its simulator dependencies installed.

The [20 curated ACMI replays](../../replays/tactical-checkpoints-v1/README.md) remain
a separate visual demonstration package. ACMI contains both aircraft, so those
replays can already be viewed from either side.
