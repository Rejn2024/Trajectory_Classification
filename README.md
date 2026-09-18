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

The lightweight tactical skill catalogue is included in this installation and
can be imported without installing the full simulator runtime or native backend:

```python
from bvr_sim.agents.skill_manager import SkillManager
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
## F-16 scripted-manoeuvre video

The runnable [`f16_scripted_manoeuvre_video.ipynb`](notebooks/f16_scripted_manoeuvre_video.ipynb)
uses JSBSim to fly a scripted F-16 demonstration and export an MP4 or GIF. See the
[setup, BVR Sim integration, validation, and troubleshooting guide](docs/f16_video_guide.md)
before running it.

To inspect a generated ACMI replay and turn it into a shareable presentation, follow the
step-by-step [Tacview visualisation and video-recording guide](docs/tacview_video_guide.md).
