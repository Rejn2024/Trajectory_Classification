# Notebook 09b diagnostic rendering and recovery

The cell beginning `DIAGNOSTICS = RUN / "diagnostics"` plots two small summaries.
It does not train, evaluate the model, scan Parquet files or upload MLflow artifacts.
The committed notebook records 60 training epochs in 7 minutes 30 seconds, followed
by final evaluation in 7.39 seconds. The plotting cell has a start timestamp of
2026-11-05 16:35:54 UTC, no completion timestamp and no displayed figures. Its
inputs are only 60 history rows and nine final summary rows. A CPU reproduction
of the original cell with representative inputs and Agg took about 0.8 seconds.

The original plotting path inherited Matplotlib's backend, called `plt.show()`
before closing each figure, and never explicitly activated inline output. On a
desktop backend, `show()` can enter a blocking GUI event loop and wait for the
window to close. This is a plausible explanation for an hours-long wait; it is
not explained by the amount of data being plotted. The saved notebook contains
neither the backend name nor an interrupt traceback, so the precise stalled call
on the original Windows kernel cannot be established from the repository alone.
Font lookup, filesystem writes and backend rendering remain other possible
stall locations. No first inline figure was recorded, so a later cell's MLflow
upload is not the cause of this cell's stall.

The notebook now selects the noninteractive Agg backend in setup and again at
the start of diagnostics, including when rerun in an existing kernel. Each figure
is laid out, rendered to PNG once, written to disk and explicitly displayed from
the same bytes. There is no `plt.show()` or desktop event loop. Fixed-size canvases
and `tight_layout()` avoid the redundant tight-bounding-box render on save, even
if a user's Matplotlib settings request one. Figures close in `finally`, including
when rendering, writing or display fails. Flushed messages identify each stage
and its elapsed time. CPU-only plotting cells no longer call the unrelated global
CUDA synchronization. Training and evaluation retain their CUDA timing barriers.

## Recover the completed run without retraining

If the original kernel still has its variables, interrupt the stuck cell, load
the updated diagnostics cell, and rerun it and the following plotting cells.
Training has already finished; avoid Run All with `TRAIN_MODEL=True`.

After a kernel restart, the first diagnostics cell can instead run entirely from
the completed run's saved tables. Execute this small setup first, substituting
the actual existing directory:

```python
from pathlib import Path
import json
import numpy as np
import pandas as pd

RUN = Path(r"C:\Users\theon\Trajectory_Classification\artifacts\paired_skill_forecasting\runs\trained-pilot-stratified-bf16_20261105T162742052741Z")
SPLITS = ("train", "test", "validation")
HORIZONS = (1, 2, 5)
history = pd.read_csv(RUN / "history.csv")
final_metrics = pd.read_csv(RUN / "final_metrics.csv")
best_epoch = json.loads((RUN / "selection.json").read_text())["epoch"]
```

Then run the cell beginning `DIAGNOSTICS = ...`. This needs no PyTorch, GPU,
prepared arrays, raw corpus downloads or MLflow connection. To regenerate the
following per-class, confusion and reliability figures too, load their saved
outputs as follows before those cells:

```python
from types import SimpleNamespace

read_json = lambda path: json.loads(Path(path).read_text())
data = SimpleNamespace(classes=read_json(RUN / "run.json")["classes"])
final_per_class = pd.read_csv(RUN / "final_per_class.csv")
final_results = {}
for split in SPLITS:
    with np.load(RUN / f"final_{split}.npz") as saved:
        final_results[split] = {key: saved[key] for key in saved.files}
MLFLOW_ENABLED = False  # Render locally; uploading is a separate optional operation.
```

The rendered PNGs remain in the run's `diagnostics/` directory and existing
training results and checkpoint selection remain usable. If rendering stalls
again, the last flushed stage identifies whether layout, PNG rendering, writing
or notebook display was reached. Interrupt the cell and retain the traceback.

## Validation

`python -m unittest discover -s tests -p test_paired_skill_diagnostics.py` executes
the actual notebook plotting cells in a subprocess with a 60-second deadline.
It starts with a different backend, makes GUI `show()` fail if called, supplies
60 epochs, all three horizons/splits, 33 classes including an absent class, and
masked calibration labels. It checks all six PNGs (including the tracked-run
comparison), their fixed dimensions, identical saved/displayed bytes, and figure
cleanup after display failure. No training data, CUDA or remote service is needed.
