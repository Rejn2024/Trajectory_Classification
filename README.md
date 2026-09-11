# BVR behaviour prediction

An implementation of the revised project plan for target-centric F-16 manoeuvre
recognition, tactical-transition forecasting, and BVR Sim native-action forecasting.

The package keeps observable features separate from privileged labels, groups splits
by episode, and exposes the simulator through a small adapter. BVR Sim itself is an
external dependency and must be pinned to commit
`db4f95657c081d120442022a339bed62d340c93b`.

```bash
python -m pip install -e '.[ml,dev]'
pytest
bvr-build-dataset --help
```

See `configs/` for the initial F-16 experiment and `examples/train_pipeline.py` for
an end-to-end training skeleton.

The executable notebook
[`notebooks/01_generate_smoke_dataset.ipynb`](notebooks/01_generate_smoke_dataset.ipynb)
walks through the first dataset milestone with a deterministic local dynamics stub. It
samples controlled scenarios, runs scripted policies through the same adapter used for
BVR Sim, writes the canonical Parquet layout, and performs schema and leakage checks.
The notebook marks the single environment-factory seam that must be replaced to run
the workflow against a pinned BVR Sim checkout.

## Stochastic skill rollouts

`StochasticSkillPolicy` drives BVR Sim native actions by instantiating persistent
skills through its `SkillManager`. Parameters and durations are sampled from bounded,
seed-replayable distributions. Geometry controls ordinary transitions, while active
missiles, low fuel, empty weapon stores, and target destruction interrupt immediately.
The selector state and transition reason are included in every generated row.

```python
from bvr_sim.agents import SkillManager
from bvr_behavior_prediction.policies import StochasticSkillPolicy
from bvr_behavior_prediction.simulator.scenario_config import ScenarioConfig

target_policy = StochasticSkillPolicy(SkillManager())
config = ScenarioConfig(backend="cpp", weapons_enabled=True)  # C++ JSBSim rollout
```

The normal sequence is `maintain_position` → `pursue_target` → `launch` →
`crank_maneuver` → `support_missile` → `turn_cold` → `recommit`. Pass a custom
`dict[str, SkillSpec]` to adjust bounds without changing selector logic.
## F-16 scripted-manoeuvre video

The runnable [`f16_scripted_manoeuvre_video.ipynb`](notebooks/f16_scripted_manoeuvre_video.ipynb)
uses JSBSim to fly a scripted F-16 demonstration and export an MP4 or GIF. See the
[setup, BVR Sim integration, validation, and troubleshooting guide](docs/f16_video_guide.md)
before running it.
