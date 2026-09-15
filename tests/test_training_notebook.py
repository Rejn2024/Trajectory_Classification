import json
import tomllib
from pathlib import Path


ROOT = Path(__file__).parents[1]
NOTEBOOK = ROOT / "notebooks" / "03_train_jsbsim_skill_classifier.ipynb"


def _notebook_source():
    notebook = json.loads(NOTEBOOK.read_text())
    return "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )


def test_pyarrow_dependency_includes_native_hotfix_release():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]

    assert "pyarrow>=14.0.1" in project["dependencies"]


def test_parquet_registry_error_has_actionable_recovery_message():
    source = _notebook_source()

    assert "except pa.ArrowKeyError as error:" in source
    assert "restart the " in source
    assert "notebook kernel, and run all cells again." in source


def test_training_uses_a_safe_cpu_default_and_allows_explicit_cuda_opt_in():
    source = _notebook_source()

    assert 'os.getenv("BVR_TRAIN_DEVICE", "cpu")' in source
    assert 'if REQUESTED_DEVICE.startswith("cuda") and not torch.cuda.is_available():' in source
    assert 'torch.device(REQUESTED_DEVICE)' in source
    assert "DEVICE = torch.device('cuda')" not in source
    assert 'DEVICE = torch.device("cuda")' not in source


def test_training_limits_torch_threads_to_avoid_notebook_resource_exhaustion():
    source = _notebook_source()

    assert 'BVR_TORCH_THREADS' in source
    assert "torch.set_num_threads(TORCH_THREADS)" in source


def test_training_stratifies_episode_skill_sets_into_60_20_20_splits():
    source = _notebook_source()

    assert 'SPLIT_FRACTIONS = {"train": 0.60, "test": 0.20, "validation": 0.20}' in source
    assert 'skills = tuple(sorted(set(episode["tactical_label"])))' in source
    assert "stratify=episode_skills[\"stratum\"]" in source
    assert "stratify=selection_episodes[\"stratum\"]" in source


def test_training_uses_transformer_and_test_selected_checkpoints():
    source = _notebook_source()

    assert "class SkillTransformer(nn.Module):" in source
    assert "nn.TransformerEncoderLayer(" in source
    assert 'test_metrics = run_epoch(loaders["test"])' in source
    assert 'if test_metrics["loss"] < best_test_loss - 1e-4:' in source
    assert 'CHECKPOINT_PATH = OUTPUT_DIR / "checkpoint.pt"' in source


def test_training_tracks_experiment_with_mlflow():
    source = _notebook_source()

    assert "import mlflow" in source
    assert "mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)" in source
    assert "with mlflow.start_run(" in source
    assert "mlflow.log_params(mlflow_params)" in source
    assert "mlflow.log_metrics(" in source
    assert "mlflow.log_artifact(" in source


def test_training_prepares_windows_without_peak_memory_copies():
    source = _notebook_source()

    assert "trajectory_dataset.scanner(" in source
    assert 'dtype=np.float32' in source
    assert 'np.lib.format.open_memmap(' in source
    assert 'np.stack(sequences)' not in source
    assert 'x[left:left + len(chunk)] -= normalization_mean' in source
    assert 'x[left:left + len(chunk)] /= normalization_scale' in source


def test_training_streams_parquet_into_disk_backed_windows():
    source = _notebook_source()

    assert ".scanner(" in source
    assert "np.lib.format.open_memmap" in source
    assert "scaler.partial_fit" in source
    assert "write_window_metadata" in source


def test_training_supports_micro_batches_and_activation_checkpointing():
    source = _notebook_source()

    assert 'BVR_MICRO_BATCH_SIZE' in source
    assert 'BVR_GRADIENT_CHECKPOINTING' in source
    assert "torch_checkpoint(" in source
    assert "ACCUMULATION_STEPS" in source


def test_cuda_training_uses_memory_saving_acceleration_paths():
    source = _notebook_source()

    assert 'AMP_ENABLED = USE_AMP and DEVICE.type == "cuda"' in source
    assert 'torch.autocast(' in source
    assert 'GradScaler(enabled=AMP_ENABLED)' in source
    assert 'fused=DEVICE.type == "cuda"' in source
    assert 'pin_memory=DEVICE.type == "cuda"' in source
    assert 'non_blocking=True' in source
    assert 'torch.inference_mode()' in source
