"""Run the notebook's actual plotting cells without a GUI, CUDA or training data."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/09b_paired_skill_forecasting.ipynb"


class DiagnosticPlotsTest(unittest.TestCase):
    def test_plotting_cells_finish_and_display_saved_pngs(self):
        # An external deadline also catches an accidental return to blocking show().
        with tempfile.TemporaryDirectory() as directory:
            environment = {**os.environ, "MPLBACKEND": "svg", "MPLCONFIGDIR": directory}
            result = subprocess.run(
                [sys.executable, str(Path(__file__).resolve()), "--render", directory],
                cwd=ROOT, env=environment, capture_output=True, text=True, timeout=60,
            )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("Rendered 6 diagnostic figures", result.stdout)

    def test_notebook_code_compiles(self):
        notebook = json.loads(NOTEBOOK.read_text())
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                source = "".join(cell["source"]).removeprefix("%%time\n")
                compile(source, str(NOTEBOOK), "exec")


def render_diagnostics(directory):
    import runpy
    from types import SimpleNamespace
    from unittest.mock import patch

    import matplotlib
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from PIL import Image as PILImage

    notebook = json.loads(NOTEBOOK.read_text())
    cells = ["".join(cell["source"]).removeprefix("%%time\n")
             for cell in notebook["cells"]]
    run = Path(directory)
    splits, horizons = ("train", "test", "validation"), (1, 2, 5)
    manager = runpy.run_path(str(ROOT / "bvr_sim_source/bvr_sim/agents/skill_manager.py"))
    classes = sorted(manager["SkillManager"]().list_skills())
    history = pd.DataFrame({
        "epoch": np.arange(1, 61), "train_nll": np.linspace(1.5, .4, 60),
        "test_nll": np.linspace(1.6, 2, 60),
        "train_macro_f1": np.linspace(.16, .56, 60),
        "test_macro_f1": np.linspace(.15, .34, 60),
    })
    metrics = pd.DataFrame([
        dict(split=s, horizon_s=h, nll=1.3, accuracy=.54, macro_f1=.25)
        for s in splits for h in horizons
    ])
    scores = pd.DataFrame([
        dict(split=s, horizon_s=h, skill=c, class_nll=1.3 if i else np.nan,
             f1=.25 if i else 0., support=10 if i else 0)
        for s in splits for h in horizons for i, c in enumerate(classes)
    ])
    targets = np.arange(64 * 3).reshape(64, 3) % len(classes)
    targets[::2, 2] = -1
    results = {
        s: dict(targets=targets, probabilities=np.full((64, 3, len(classes)),
                                                      1 / len(classes)),
                confusion=np.stack([np.diag([0] + [10] * (len(classes) - 1))] * 3))
        for s in splits
    }
    namespace = dict(
        RUN=run, history=history, final_metrics=metrics, best_epoch=9,
        final_per_class=scores, final_results=results, SPLITS=splits,
        HORIZONS=horizons, data=SimpleNamespace(classes=classes),
        plt=plt, np=np, pd=pd, MLFLOW_ENABLED=False,
    )  # Deliberately no torch, training or raw-corpus dependency.
    displayed = []
    import IPython.display

    with patch.object(plt, "show", side_effect=AssertionError("GUI show must not be used")), \
            patch.object(IPython.display, "display", side_effect=displayed.append), \
            matplotlib.rc_context({"savefig.bbox": "tight"}):
        for index in (12, 13, 14):
            exec(compile(cells[index], f"diagnostics-cell-{index}", "exec"), namespace)
        # Exercise the shared helper in the tracked-run comparison too, offline.
        namespace.update(
            MLFLOW_ENABLED=True, TRACKING={}, DATA=run,
            sha256=lambda path: "fixture",
            compare_runs=lambda *args: pd.DataFrame([
                dict(run_id="fixture", name="fixture", best_test_nll=1.3)]),
            metric_history=lambda *args: pd.DataFrame(dict(epoch=[1, 2], value=[1.4, 1.3])),
            log_run_artifacts=lambda path: None,
        )
        exec(compile(cells[17], "diagnostics-comparison", "exec"), namespace)
        assert matplotlib.get_backend().lower() == "agg"
        assert plt.get_fignums() == []

        names = ["learning_curves", "final_split_comparison", "per_class_f1_and_support",
                 "confusion_matrices", "reliability", "mlflow_run_comparison"]
        sizes = [(2080, 640), (2560, 640), (3200, 2240), (3840, 3680),
                 (2560, 640), (2240, 640)]
        pngs = [obj for obj in displayed if isinstance(obj, IPython.display.Image)]
        assert len(pngs) == len(names)
        for name, size, displayed_png in zip(names, sizes, pngs):
            path = run / "diagnostics" / f"{name}.png"
            assert path.read_bytes() == displayed_png.data
            with PILImage.open(path) as saved:
                assert saved.size == size  # No tight-bbox second render or unbounded canvas.
                saved.verify()

        # A failed notebook display must still release the large Matplotlib figure.
        def fail_display(obj):
            raise RuntimeError("display failed")

        with patch.dict(namespace, display=fail_display):
            fig, _ = plt.subplots()
            try:
                namespace["show_figure"](fig, "failed_display")
            except RuntimeError as error:
                assert str(error) == "display failed"
            else:
                raise AssertionError("Expected the display error")
            assert plt.get_fignums() == []
    print(f"Rendered {len(pngs)} diagnostic figures")


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--render":
        render_diagnostics(sys.argv[2])
    else:
        unittest.main()
