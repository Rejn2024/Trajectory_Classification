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

