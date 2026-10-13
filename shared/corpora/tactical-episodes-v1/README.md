# Tactical opponent episode dataset

Raw encounter data for forecasting the scripted tactical opponent from a trained
pilot's perspective. The collection contains 300 episodes for each of ten frozen
tactical-adapted pilots, giving 3,000 encounters. Both aircraft are F-16s.

The observation stream preserves the exact 51 values supplied by the existing
pilot environment at each simulator sample. Aircraft tracks in physical units
are provided alongside it. No manoeuvre labels, future prediction targets,
history windows, train/test assignments or additional fitted normalisation are
included. Those preprocessing choices remain open.

## Contents

| File or directory | Contents |
| --- | --- |
| `observations/pilot_NNN.parquet` | 51 float32 observation values at 10 Hz, plus episode ID, step and time |
| `tracks/pilot_NNN.parquet` | Observer and target positions, velocities, alive flags and cumulative launch counts at the same timestamps |
| `decisions/pilot_NNN.parquet` | Learned pilot's selected skill and physical parameters, recorded at the decision time |
| `episodes.parquet` | Episode ID, pilot, scenario group, seed, checkpoint hash, duration, actual initial state and final outcome |
| `config.json` | Requested scenarios, checkpoint hashes, frozen source hashes and collection settings |
| `schema.json` | Column keys and storage conventions |
| `metrics.json` | Collection counts and validation checks |
| `manifest.json` | File sizes, SHA-256 checksums, software versions and provenance |
| `reproduction.zip` | Frozen simulator and policy source, assets, and ten selected model checkpoints |

Each table has one compressed shard per pilot, with one Parquet row group per
episode. `episode_id`, `step` and `time_s` align observations with tracks.
Decisions are recorded less frequently, normally every second. The first row is
the reset state at time zero; subsequent rows follow each 0.1-second simulator
step, including the terminal sample. Durations are at most 120 seconds, and many
encounters finish earlier.

The observer is **A01**, the learned pilot. The prediction target is **B01**, the
scripted tactical opponent. Native simulator colours are Red for A01 and Blue for
B01; the RL adapter uses the opposite team names. Identify roles by aircraft ID.

## Download

From a checkout of the `multi-pilot-training` branch with Git LFS installed:

```powershell
git lfs pull --include="shared/corpora/tactical-episodes-v1/**" --exclude=""
```

To download one pilot and the episode index first:

```powershell
git lfs pull --include="shared/corpora/tactical-episodes-v1/*/pilot_001.parquet,shared/corpora/tactical-episodes-v1/episodes.parquet" --exclude=""
```

Parquet files already use Zstandard compression. They are stored directly in LFS,
without a second full-dataset ZIP that would duplicate storage. Small JSON metadata
and this README remain ordinary Git files. Read the actual compressed byte sizes
in [the manifest](manifest.json).

GitHub Free and Pro allow LFS files up to 2 GB; this exporter enforces a more
conservative 1 GB cap per published file. Downloads consume the repository owner's
LFS bandwidth, and replacing an LFS file stores a new complete version. Use
selective downloads and publish revised corpora under a new version. Account
allowances are shared across repositories; this manifest does not establish the
owner's remaining quota. See [GitHub file limits](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-git-large-file-storage)
and [GitHub LFS usage](https://docs.github.com/en/billing/concepts/product-billing/git-lfs).

## Read an episode

No notebook or simulation is required to read the published data. Python with
`pandas` and `pyarrow` is sufficient.

```python
from pathlib import Path
import pandas as pd

root = Path("shared/corpora/tactical-episodes-v1")
episodes = pd.read_parquet(root / "episodes.parquet")
episode_id = "pilot_001_case_0000"
filters = [("episode_id", "==", episode_id)]
observations = pd.read_parquet(root / "observations/pilot_001.parquet", filters=filters)
tracks = pd.read_parquet(root / "tracks/pilot_001.parquet", filters=filters)
decisions = pd.read_parquet(root / "decisions/pilot_001.parquet", filters=filters)
print(episodes.loc[episodes.episode_id == episode_id])
print(observations.head())
```

## Observation definition

The `obs_00` to `obs_50` columns retain the controller's existing fixed scales;
they are not an unscaled sensor measurement vector. No additional scaling was
fitted to this dataset. Coordinates in `tracks/` use North, West and Up, in metres;
velocity components use metres per second. Relative observation position and
velocity components use the same world axes, not rotating aircraft body axes.

| Columns | Meaning | Existing scale |
| --- | --- | --- |
| `obs_00`–`obs_02` | Observer position north, west, up | Horizontal / 50,000 m; vertical / 10,000 m |
| `obs_03`–`obs_05` | Observer velocity north, west, up | / 600 m/s |
| `obs_06` | Observer altitude | / 10,000 m |
| `obs_07`–`obs_08` | Observer heading sine and cosine | Unitless |
| `obs_09`–`obs_11` | Opponent position relative to observer | Horizontal / 50,000 m; vertical / 10,000 m |
| `obs_12`–`obs_14` | Opponent velocity relative to observer | / 600 m/s |
| `obs_15` | Opponent range | / 50,000 m |
| `obs_16`–`obs_17` | Opponent horizontal bearing relative to observer heading, sine and cosine | Unitless |
| `obs_18` | Incoming missile indicator | 0 or 1 |
| `obs_19`–`obs_46` | Up to four live friendly missile slots, seven values each: relative position, relative velocity and range | Position scales as above; velocity / 1,200 m/s; range / 50,000 m |
| `obs_47` | Observer speed-derived kinetic energy proxy | (speed / 400 m/s) squared |
| `obs_48` | Observer altitude-derived potential energy proxy | altitude / 12,000 m |
| `obs_49` | Opponent speed-derived kinetic energy proxy | (speed / 400 m/s) squared |
| `obs_50` | Opponent altitude-derived potential energy proxy | altitude / 12,000 m |

Missing missile slots are zero-filled. The opponent block is zeroed when that
aircraft is no longer alive, and the compact observation is zeroed if the observer
is no longer alive. The four appended energy values follow the adapter's own
state queries. Preserve alive flags when choosing usable samples later.
Missile slots compact the remaining live missiles; slot numbers are not stable
missile identities.

The controller consumes a history of 20 frames. The dataset stores each actual
frame once rather than saving overlapping windows or duplicating the controller's
initial left padding. A later forecasting model can use any causal history length,
including the complete preceding encounter. Initialisation transients remain
present, as emitted by the simulator.

## Scenario design and limitations

The ten pilots encounter the same 300 fresh scenario seeds, from 3,000,000 to
3,000,299. This produces 3,000 encounters from **300 matched starting scenarios**,
not 3,000 independent initial geometries. There are 100 requested starting ranges
at each of 18, 30 and 45 km. All completed outcomes are retained; this corpus is
not selected for victories or visual interest.

The existing scenario sampler and simulator are unchanged. The adapter applies
separation and an altitude interval; the native spawn manager samples the
engagement axis and aircraft altitudes, with approximately opposing initial
courses and a fixed initial speed. The nominal scenario bearing, individual
headings and speeds are not directly applied. Use `initial_state_json` in the
episode index for the realised starting conditions.

The compact observation interface obtains exact opponent kinematics from the
simulator. These are idealised tracks from the observer's interface, without
radar noise, occlusion or track-loss modelling. The corpus contains one scripted
tactical target-controller family. The ten learned policies provide different
observer behaviour, not ten independently learned target policies.

No data split is imposed. When preprocessing later, keep all encounters sharing
`scenario_group` in the same partition, and form windows within individual
episodes. At prediction time, inputs should contain only observations at or before
that time. Final outcomes and later observations are metadata or future data,
not forecasting inputs.

## Collection and verification

The collector uses deterministic inference from each completed tactical run's
validation-selected `best_model.pt`. It does not update model weights or run PPO.
Checkpoint hashes and the frozen simulator snapshot identify the exact inputs.
Model and simulator arithmetic can differ slightly across hardware or batching.

```powershell
.venv\Scripts\python.exe -X utf8 scripts/generate_tactical_corpus.py verify
```

To generate an independent copy on a machine configured with this repository's
simulation dependencies, extract the reproduction bundle and choose a fresh local
working directory:

```powershell
Expand-Archive shared/corpora/tactical-episodes-v1/reproduction.zip -DestinationPath artifacts/tactical-corpus-source
.venv\Scripts\python.exe -X utf8 scripts/generate_tactical_corpus.py collect --job artifacts/tactical-corpus-source --work artifacts/tactical-corpus-copy --workers 8
.venv\Scripts\python.exe -X utf8 scripts/generate_tactical_corpus.py package --job artifacts/tactical-corpus-source --work artifacts/tactical-corpus-copy --output artifacts/tactical-corpus-packaged
```

The collector supports resuming the same unchanged configuration. Progress is
written to `progress.json` in its working directory. The default local collection
directory is `artifacts/tactical_episode_data_v1`. Failed episodes are reported
with their original seeds rather than replaced with easier cases.

For a count taken directly from completed episode files:

```powershell
.venv\Scripts\python.exe -X utf8 scripts/generate_tactical_corpus.py status
```

The [20 selected-checkpoint ACMI replays](../../replays/tactical-checkpoints-v1/README.md)
are a separate curated demonstration package. They are not the 3,000 corpus
episodes and should not be used to estimate corpus outcome frequencies.
