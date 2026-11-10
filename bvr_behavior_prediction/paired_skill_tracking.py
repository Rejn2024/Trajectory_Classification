"""MLflow tracking for paired forecasting, without changing fluent/global run state."""

import hashlib
import json
import math
import re
import shutil
import tempfile
import time
import warnings
from pathlib import Path


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _value(value):
    return json.dumps(value, sort_keys=True) if isinstance(value, (list, tuple, dict)) else str(value)


class RunTracker:
    def __init__(self, output, settings=None):
        self.output, self.client, self.state = Path(output), None, None
        self.link = self.output / "mlflow.json"
        self.previous = _read(self.link) if self.link.exists() else None
        if settings is None and self.previous is not None:
            settings = self.previous["settings"]
        self.settings = {"enabled": False, "experiment": "paired-skill-forecasting",
                         "log_per_class": True, "log_model": True, **(settings or {})}
        self.enabled = self.settings["enabled"]
        self.phase = "running"

    def __enter__(self):
        if not self.enabled:
            return self
        try:
            from mlflow.tracking import MlflowClient
        except ImportError as error:
            raise RuntimeError("MLflow is enabled; install the project's .[ml] dependencies "
                               "or set MLFLOW_ENABLED=False.") from error
        uri = str(self.settings.get("tracking_uri", ""))
        if not uri:
            raise ValueError("An explicit MLflow tracking_uri is required")
        self.settings["tracking_uri"] = uri
        self.client = MlflowClient(tracking_uri=uri)
        if self.previous is not None:
            old = self.previous["settings"]
            if uri != old["tracking_uri"] or self.settings["experiment"] != old["experiment"]:
                raise ValueError("A resumed run must use its existing MLflow store and experiment")
            self.state = self.previous
            self.client.get_run(self.state["run_id"])
            self.client.update_run(self.state["run_id"], status="RUNNING")
        else:
            experiment = self.client.get_experiment_by_name(self.settings["experiment"])
            if experiment is None:
                location = self.settings.get("artifact_root")
                try:
                    experiment_id = self.client.create_experiment(
                        self.settings["experiment"], artifact_location=location)
                except Exception:
                    experiment = self.client.get_experiment_by_name(self.settings["experiment"])
                    if experiment is None:
                        raise
                    experiment_id = experiment.experiment_id
            else:
                experiment_id = experiment.experiment_id
            active = self.client.create_run(experiment_id, tags={
                "mlflow.runName": self.settings.get("run_name", self.output.name),
                "pipeline": "paired_skill_forecasting",
                "test_role": "checkpoint_selection", "validation_role": "held_out_until_final",
            })
            self.state = {"run_id": active.info.run_id, "experiment_id": experiment_id,
                          "artifact_uri": active.info.artifact_uri, "settings": self.settings,
                          "last_logged_epoch": 0, "model_checkpoint_sha256": None}
        self.state["settings"] = self.settings
        _write(self.link, self.state)
        self.client.set_tag(self.state["run_id"], "phase", "running")
        print(f"MLflow run: {self.state['run_id']} ({self.settings['experiment']})", flush=True)
        return self

    def __exit__(self, exc_type, exc, traceback):
        if not self.enabled or self.client is None:
            return False
        status = "FINISHED" if exc_type is None else (
            "KILLED" if issubclass(exc_type, (KeyboardInterrupt, SystemExit)) else "FAILED")
        try:
            if exc_type is not None:
                self.client.set_tag(self.state["run_id"], "failure_type", exc_type.__name__)
                self.client.set_tag(self.state["run_id"], "failure_message", str(exc)[:2000])
            self.client.set_tag(self.state["run_id"], "phase", self.phase if exc_type is None else "interrupted")
            self.client.set_terminated(self.state["run_id"], status=status)
        except Exception:
            if exc_type is None:
                raise
            warnings.warn("Could not update MLflow status; preserving the original exception.")
        return False

    def metrics(self, values, step):
        if not self.enabled:
            return
        from mlflow.entities import Metric
        timestamp = int(time.time() * 1000)
        metrics = [Metric(str(key), float(value), timestamp, int(step)) for key, value in values.items()
                   if math.isfinite(float(value))]
        for start in range(0, len(metrics), 1000):
            self.client.log_batch(self.state["run_id"], metrics=metrics[start:start + 1000])

    def artifact(self, path, directory):
        if self.enabled and Path(path).is_file():
            self.client.log_artifact(self.state["run_id"], str(path), directory)

    def begin(self, run, data):
        if not self.enabled:
            return
        from mlflow.entities import Param
        from .paired_skill_forecasting import DATA_CONTRACT
        self.classes = list(run["classes"])
        contract = data.manifest.get("data_contract", run.get("data_contract", DATA_CONTRACT))
        self.client.set_tag(self.state["run_id"], "data_contract", contract)
        self.client.set_tag(self.state["run_id"], "architecture", run.get("architecture", "transformer"))
        self.client.set_tag(self.state["run_id"], "pipeline",
                            "trained_pilot_skill_forecasting" if contract == DATA_CONTRACT else contract)
        if self.previous is None:
            params = {f"training.{k}": _value(v) for k, v in run["training"].items()}
            params.update(dataset_manifest_sha256=run["data_manifest_sha256"],
                          source_sha256=run["source_sha256"], data_contract=contract,
                          tracking_source_sha256=_sha(__file__),
                          history_steps=51, raw_history_steps=56, input_features=len(data.manifest["features"]),
                          horizons_s="1,2,5", classes=len(self.classes),
                          split_fractions="train=0.4,test=0.3,validation=0.3",
                          selection="strict_minimum_mean_test_NLL")
            self.client.log_batch(self.state["run_id"], params=[Param(k, _value(v)) for k, v in params.items()])
        for name in ("manifest.json", "scaling.json", "scenario_splits.json", "split_assignments.csv",
                     "skill_support_before_split.csv", "skill_support_by_split.csv",
                     "window_sampling_audit.csv", "nonoverlap_audit.json", "enemy_missile_audit.csv"):
            self.artifact(data.directory / name, "data_metadata")
        for name in ("intent_policy.json", "skill_intent_mapping.csv", "intent_support_by_split.csv"):
            self.artifact(data.directory / name, "data_metadata")
        for name in ("skill_hierarchy.json", "skill_family_direction_mapping.csv",
                     "family_support_by_split.csv", "direction_support_by_split.csv"):
            self.artifact(data.directory / name, "data_metadata")
        self.artifact(self.output / "training_source.py", "source")
        self.artifact(self.output / "architecture_source.py", "source")
        self.artifact(self.output / "intent_source.py", "source")
        shutil.copy2(__file__, self.output / "tracking_source.py")
        self.artifact(self.output / "tracking_source.py", "source")
        self.artifact(self.output / "run.json", "metadata")
        for name in ("class_weights.json", "class_weights.csv", "handedness_weights.json", "handedness_weights.csv"):
            self.artifact(self.output / name, "metadata")
        if self.settings.get("benchmark_path"):
            self.artifact(self.settings["benchmark_path"], "benchmarks")

    def runtime(self, run):
        if not self.enabled:
            return
        import os
        import platform
        import torch
        for key in ("device", "torch_version", "numpy_version", "effective_precision",
                    "effective_data_backend", "effective_cpu_threads"):
            self.client.set_tag(self.state["run_id"], f"runtime.{key}", str(run[key]))
        for key in ("effective_evaluation_precision", "evaluation_attention"):
            if key in run:
                self.client.set_tag(self.state["run_id"], f"runtime.{key}", str(run[key]))
        hardware = {"python": platform.python_version(), "cpu": platform.processor(),
                    "logical_cpus": os.cpu_count(), "cuda_runtime": torch.version.cuda}
        if torch.device(run["device"]).type == "cuda":
            gpu = torch.cuda.get_device_properties(torch.device(run["device"]))
            hardware.update(gpu=gpu.name, gpu_memory_GiB=gpu.total_memory / 2**30)
        for key, value in hardware.items():
            self.client.set_tag(self.state["run_id"], f"hardware.{key}", str(value))
        self.metrics({"runtime/frame_cache_GiB": run["frame_cache_bytes"] / 2**30}, 0)
        self.artifact(self.output / "run.json", "metadata")

    def table_metrics(self, summary, per_class, prefix):
        result = {}
        for row in summary.to_dict("records"):
            split, horizon = row["split"], row["horizon_s"]
            for key in ("count", "nll", "accuracy", "macro_f1", "supported_macro_f1", "weighted_f1"):
                if key in row:
                    result[f"{prefix}/{split}/{horizon:g}s/{key}"] = row[key]
        if self.settings["log_per_class"]:
            mapping = {name: i for i, name in enumerate(self.classes)}
            for row in per_class.to_dict("records"):
                name = re.sub(r"[^A-Za-z0-9_.-]", "_", str(row["skill"]))[:80]
                label = f"class_{mapping[row['skill']]:02d}_{name}"
                for key in ("support", "f1", "class_nll"):
                    result[f"{prefix}/{row['split']}/{row['horizon_s']:g}s/{label}/{key}"] = row[key]
        return result

    def epoch(self, row, summary, per_class):
        if not self.enabled or int(row["epoch"]) <= self.state["last_logged_epoch"]:
            return
        metrics = self.table_metrics(summary, per_class, "epoch")
        metrics.update({"epoch/train/nll": row["train_nll"], "epoch/test/nll": row["test_nll"],
                        "epoch/train/macro_f1": row["train_macro_f1"],
                        "epoch/test/macro_f1": row["test_macro_f1"],
                        "epoch/fit_and_evaluation_seconds": row["seconds"],
                        "epoch/epochs_without_improvement": row["epochs_without_improvement"]})
        for key in ("train_data_loss", "train_l1_penalty", "train_objective", "generalization_gap",
                    "overtraining_epochs", "overtraining_detected", "learning_rate", "learning_rate_start",
                    "next_learning_rate", "optimizer_steps", "train_skill_data_loss", "train_intent_nll",
                    "test_intent_nll", "train_intent_accuracy", "test_intent_accuracy", "train_family_nll",
                    "test_family_nll", "train_direction_nll", "test_direction_nll", "train_hierarchical_credit",
                    "test_hierarchical_credit", "train_expected_hierarchical_credit", "test_expected_hierarchical_credit",
                    "train_family_data_loss", "train_handedness_data_loss"):
            if key in row:
                metrics[f"epoch/{key}"] = row[key]
        if row["improved"]:
            metrics.update({"best/test_nll": row["test_nll"], "best/epoch": row["epoch"]})
        self.metrics(metrics, row["epoch"])
        self.state["last_logged_epoch"] = int(row["epoch"])
        _write(self.link, self.state)

    def backfill(self):
        """Recover completed local epochs after an interrupted MLflow write."""
        if not self.enabled or not (self.output / "history.csv").exists():
            return
        import pandas as pd
        for row in pd.read_csv(self.output / "history.csv").to_dict("records"):
            if row["epoch"] > self.state["last_logged_epoch"]:
                stem = self.output / "epochs" / f"epoch_{int(row['epoch']):03d}"
                self.epoch(row, pd.read_csv(str(stem) + "_summary.csv"),
                           pd.read_csv(str(stem) + "_per_class.csv"))

    def finish_training(self):
        if not self.enabled:
            return
        status = _read(self.output / "training_status.json")
        self.metrics({"training/completed_epochs": status["completed_epochs"],
                      "training/early_stopped": int(status["status"] == "early_stopped"),
                      "best/epoch": status["best_epoch"], "best/test_nll": status["best_test_nll"]},
                     status["completed_epochs"])
        self.client.set_tag(self.state["run_id"], "training_status", status["status"])
        if status.get("stop_reason"):
            self.client.set_tag(self.state["run_id"], "stop_reason", status["stop_reason"])
        self.sync_artifacts()
        self.save_model()
        self.phase = "training_complete_validation_pending"

    def final(self, epoch, summary, per_class):
        if not self.enabled:
            return
        self.classes = _read(self.output / "run.json")["classes"]
        metrics = self.table_metrics(summary, per_class, "final")
        for split, rows in summary.groupby("split"):
            for key in ("nll", "accuracy", "macro_f1", "supported_macro_f1", "weighted_f1"):
                metrics[f"final/{split}/{key}"] = float(rows[key].mean())
        metrics["final/selected_epoch"] = epoch
        self.metrics(metrics, epoch)
        self.sync_artifacts()
        self.phase = "final_evaluated"

    def sync_artifacts(self):
        if not self.enabled:
            return
        for name in ("run.json", "training_status.json", "selection.json", "mlflow.json",
                     "nonfinite_evaluation.json"):
            self.artifact(self.output / name, "metadata")
        for name in ("history.csv", "final_metrics.csv", "final_per_class.csv", "training_frequency_baseline.csv", "mlflow_comparison.csv"):
            self.artifact(self.output / name, "metrics")
        for name in ("final_intent_metrics.csv", "final_intent_per_class.csv", "training_frequency_intent_baseline.csv"):
            self.artifact(self.output / name, "metrics")
        for name in ("final_family_metrics.csv", "final_family_per_class.csv", "final_direction_metrics.csv",
                     "final_direction_per_class.csv", "final_hierarchical_credit_metrics.csv"):
            self.artifact(self.output / name, "metrics")
        self.artifact(self.output / "best.pt", "checkpoints")
        for name in ("epochs", "diagnostics", "intent_diagnostics"):
            if (self.output / name).is_dir():
                self.client.log_artifacts(self.state["run_id"], str(self.output / name), name)

    def save_model(self):
        if not self.enabled or not self.settings["log_model"]:
            return
        checkpoint = self.output / "best.pt"
        digest = _sha(checkpoint)
        if self.state["model_checkpoint_sha256"] == digest:
            return
        import mlflow
        import mlflow.pytorch
        import inspect
        import numpy as np
        import pandas as pd
        import pyarrow
        import torch
        from mlflow.models import ModelSignature
        from mlflow.types import Schema, TensorSpec
        from .paired_skill_forecasting import make_model

        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        input_features = len(saved["feature_names"])
        with torch.random.fork_rng(devices=[]):
            model = make_model(len(saved["classes"]), saved["model_config"], saved["mean"], saved["std"])
            model.load_state_dict(saved["model_state"])
            model.eval()
        with tempfile.TemporaryDirectory(prefix="paired-mlflow-") as temporary:
            root = Path(temporary)
            package = root / "source" / "bvr_behavior_prediction"
            package.mkdir(parents=True)
            for name in ("__init__.py", "paired_skill_forecasting.py", "paired_skill_tracking.py",
                         "enemy_missile_models.py"):
                shutil.copy2(Path(__file__).parent / name, package / name)
            signature = ModelSignature(
                inputs=Schema([TensorSpec(np.dtype("float32"), (-1, 51, input_features))]),
                outputs=Schema([TensorSpec(np.dtype("float32"), (-1, 3, len(saved["classes"]))) ]))
            requirements = [f"mlflow=={mlflow.__version__}", f"torch=={str(torch.__version__).split('+')[0]}",
                            f"numpy=={np.__version__}", f"pandas=={pd.__version__}", f"pyarrow=={pyarrow.__version__}"]
            # New MLflow defaults to torch.export. Preserve dynamic batch sizes with
            # an example batch > 1; older MLflow retains its native eager format.
            options = {"code_paths": [str(package)]}
            export_format = "pickle"
            if "serialization_format" in inspect.signature(mlflow.pytorch.save_model).parameters:
                major, minor = [int(v) for v in str(torch.__version__).split("+")[0].split(".")[:2]]
                if (major, minor) >= (2, 4):
                    export_format = "pt2"
                    options = {"serialization_format": "pt2", "input_example": np.zeros((2, 51, input_features), dtype=np.float32)}
                else:
                    options["serialization_format"] = "pickle"
            mlflow.pytorch.save_model(model, str(root / "model"), signature=signature,
                                      pip_requirements=requirements, **options)
            if export_format == "pt2":
                shutil.copytree(root / "source", root / "model" / "source_snapshot")
            _write(root / "model" / "paired_input_schema.json", {
                "input": f"Normalized feature histories, shape [batch,51,{input_features}]",
                "output": ("Joint original-skill log probabilities, shape [batch,3,classes]; exp or softmax on last axis"
                           if saved["model_config"].get("architecture") == "hgt_hierarchical_intent" else
                           "Raw logits, shape [batch,3,classes]; apply softmax on the last axis"),
                "skill_hierarchy": saved.get("skill_hierarchy"),
                "classes": saved["classes"], "features": saved["feature_names"],
                "architecture": saved["model_config"].get("architecture", "transformer"),
                "data_contract": saved.get("data_contract"),
                "input_contract": saved.get("input_contract"),
                "target_contract": saved.get("target_contract"),
                "mean": saved["mean"].tolist(), "std": saved["std"].tolist(),
                "horizons_s": saved["horizons_s"], "selected_epoch": saved["epoch"],
                "reported_evaluation_precision": saved.get("effective_evaluation_precision",
                                                            saved.get("effective_precision", "fp32")),
                "evaluation_attention": saved.get("evaluation_attention", "auto"),
                "export_precision": "FP32; use matching CUDA autocast to reproduce BF16 evaluation",
                "serialization_format": export_format,
            })
            self.client.log_artifacts(self.state["run_id"], str(root / "model"), "model")
        self.state["model_checkpoint_sha256"] = digest
        self.state["model_uri"] = f"runs:/{self.state['run_id']}/model"
        _write(self.link, self.state)
        self.client.set_tag(self.state["run_id"], "model_uri", self.state["model_uri"])
        self.artifact(self.link, "metadata")


def log_run_artifacts(output):
    """Attach figures/baseline tables produced after final_evaluation to the same run."""
    with RunTracker(output) as tracker:
        tracker.sync_artifacts()
        tracker.phase = "diagnostics_complete"


def compare_runs(settings, dataset_manifest_sha256, max_results=50):
    import pandas as pd
    from mlflow.tracking import MlflowClient
    if not re.fullmatch(r"[a-f0-9]{64}", dataset_manifest_sha256):
        raise ValueError("Expected a dataset SHA-256 digest")
    client = MlflowClient(tracking_uri=settings["tracking_uri"])
    experiment = client.get_experiment_by_name(settings["experiment"])
    if experiment is None:
        return pd.DataFrame()
    runs = client.search_runs([experiment.experiment_id],
                              filter_string=f"params.dataset_manifest_sha256 = '{dataset_manifest_sha256}'",
                              order_by=["start_time DESC"], max_results=max_results)
    rows = []
    for run in runs:
        params, metrics, tags = run.data.params, run.data.metrics, run.data.tags
        row = {"run_id": run.info.run_id, "name": tags.get("mlflow.runName"), "status": run.info.status,
               "phase": tags.get("phase"), "training_status": tags.get("training_status"),
               "batch_size": params.get("training.batch_size"), "heads": params.get("training.heads"),
               "width": params.get("training.width"), "layers": params.get("training.layers"),
               "learning_rate": params.get("training.learning_rate"), "seed": params.get("training.seed"),
               "precision": tags.get("runtime.effective_precision"),
               "backend": tags.get("runtime.effective_data_backend"),
               "device": tags.get("runtime.device"), "gpu": tags.get("hardware.gpu"),
               "compiled": params.get("training.compile_model"),
               "epochs": metrics.get("training/completed_epochs"), "best_epoch": metrics.get("best/epoch"),
               "best_test_nll": metrics.get("best/test_nll"), "model_uri": tags.get("model_uri")}
        for split in ("train", "test", "validation"):
            for key in ("nll", "accuracy", "macro_f1"):
                row[f"final_{split}_{key}"] = metrics.get(f"final/{split}/{key}")
        rows.append(row)
    return pd.DataFrame(rows)


def metric_history(settings, run_id, key):
    import pandas as pd
    from mlflow.tracking import MlflowClient
    values = MlflowClient(tracking_uri=settings["tracking_uri"]).get_metric_history(run_id, key)
    return pd.DataFrame([{"epoch": m.step, "value": m.value, "timestamp": m.timestamp}
                         for m in values]).sort_values(["epoch", "timestamp"]) if values else pd.DataFrame()
