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
    assert (
        'split_audit = pd.concat([\n'
        '    table.assign(split=name).explode("skills")\n'
        '    for name, table in split_tables.items()\n'
        '], ignore_index=True)'
    ) in source


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
    assert 'REPO_ROOT / "artifacts/mlflow.db"' in source
    assert 'f"sqlite:///{MLFLOW_DB_PATH.as_posix()}"' in source
    assert '(REPO_ROOT / "artifacts/mlruns").resolve().as_uri()' not in source
    assert "mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)" in source
    assert "with mlflow.start_run(" in source
    assert "mlflow.log_params(mlflow_params)" in source
    assert "mlflow.log_metrics(" in source
    assert "mlflow.log_artifact(" in source


def test_training_prepares_windows_with_ram_accelerated_serialization():
    source = _notebook_source()

    assert "dataset.scanner(" in source
    assert 'dtype=np.float32' in source
    assert 'np.empty(shape, dtype=dtype)' in source
    assert 'np.save(path, array, allow_pickle=False)' in source
    assert 'np.stack(sequences)' not in source
    assert "chunk -= normalization_mean" in source
    assert "chunk /= normalization_scale" in source


def test_training_indexes_windows_without_materializing_overlapping_features():
    source = _notebook_source()

    assert 'WINDOW_CACHE_VERSION = 2' in source
    assert '"representation": "trajectory_samples_with_window_offsets"' in source
    assert 'destination["window_start"][window_left:window_right] = sample_left + starts' in source
    assert 'destination["features"][sample_left:sample_right] = np.column_stack(' in source
    assert 'sliding_window_view' not in source
    assert 'TensorDataset' not in source


def test_lazy_window_dataset_resolves_compact_offsets_on_demand():
    source = _notebook_source()

    assert 'class TrajectoryWindowDataset(torch.utils.data.Dataset):' in source
    assert 'start = int(self.window_start[index])' in source
    assert 'self.features[start:start + self.window_samples]' in source
    assert 'dataset = TrajectoryWindowDataset(values, WINDOW_SAMPLES)' in source

def test_training_streams_parquet_into_persisted_disk_backed_windows():
    source = _notebook_source()

    assert ".scanner(" in source
    assert "np.save(path, array, allow_pickle=False)" in source
    assert 'np.load(CACHE_DIR / f"{split_name}_{suffix}.npy", mmap_mode="r+")' in source
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


def test_metadata_scan_is_parallel_bounded_and_persistently_cached():
    notebook = json.loads(NOTEBOOK.read_text())
    source = _notebook_source()

    assert "ThreadPoolExecutor(max_workers=worker_count)" in source
    assert "executor.map(scan_metadata_shard, shards)" in source
    assert '"BVR_METADATA_SCAN_WORKERS"' in source
    assert '"fingerprint": metadata_fingerprint' in source
    assert "os.replace(temporary_cache_path, METADATA_CACHE_PATH)" in source

    scan_cell_index = next(
        index for index, cell in enumerate(notebook["cells"])
        if cell["cell_type"] == "code"
        and "shards = sorted((DATASET_DIR / \"trajectories\").glob(\"*.parquet\"))"
        in "".join(cell.get("source", []))
    )
    reload_source = "".join(notebook["cells"][scan_cell_index + 1].get("source", []))
    assert "reloaded_metadata = load_metadata_cache()" in reload_source
    assert 'reloaded_metadata["episode_rows"] == episode_rows' in reload_source


def test_metadata_cache_documents_speed_and_reports_elapsed_time():
    notebook = json.loads(NOTEBOOK.read_text())
    markdown = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "markdown"
    )
    source = _notebook_source()

    assert "compact `episodes.parquet` index plus Parquet footer" in markdown
    assert "10–100× or more" in markdown
    assert "time.perf_counter()" in source
    assert "metadata_elapsed_s" in source


def test_metadata_uses_compact_episode_index_and_parquet_footers_first():
    source = _notebook_source()

    assert 'import pyarrow.parquet as pq' in source
    assert 'def load_compact_metadata():' in source
    assert 'episode_file.iter_batches(' in source
    assert 'batch_size=METADATA_BATCH_ROWS' in source
    assert 'executor.map(parquet_row_count, shards)' in source
    assert 'first_file.iter_batches(batch_size=2, columns=["time_s"])' in source
    assert '"BVR_FORCE_METADATA_SCAN"' in source
    assert 'compact episode index + Parquet footers' in source


def test_compact_metadata_path_keeps_parallelism_and_memory_bounded():
    source = _notebook_source()

    assert '"BVR_METADATA_BATCH_ROWS"' in source
    assert "if METADATA_BATCH_ROWS < 1:" in source
    assert "for batch in episode_file.iter_batches(" in source
    assert "columns = batch.to_pydict()" in source
    assert "for episode_id, episode_duration_s, schedule_json in records:" in source
    assert "episode_table.to_pylist()" not in source
    assert "read_row_group(0" not in source


def test_metadata_fingerprint_invalidates_when_episode_index_changes():
    source = _notebook_source()

    assert '"episodes": file_identity(episodes_path)' in source
    assert 'METADATA_CACHE_VERSION = 3' in source


def test_window_products_are_parallel_counted_and_persistently_cached():
    notebook = json.loads(NOTEBOOK.read_text())
    source = _notebook_source()

    assert "executor.map(prepare_window_shard, shards)" in source
    assert 'WINDOW_CACHE_MANIFEST = CACHE_DIR / "manifest.json"' in source
    assert '"fingerprint": window_cache_fingerprint' in source
    assert 'np.load(CACHE_DIR / f"{split_name}_{suffix}.npy", mmap_mode="r+")' in source
    assert "os.replace(temporary_manifest, WINDOW_CACHE_MANIFEST)" in source
    assert 'BVR_KEEP_WINDOW_CACHE", "1"' in source

    cache_cell_index = next(
        index for index, cell in enumerate(notebook["cells"])
        if cell["cell_type"] == "code"
        and "episode_to_split = {" in "".join(cell.get("source", []))
    )
    reload_source = "".join(notebook["cells"][cache_cell_index + 1].get("source", []))
    assert "reloaded_window_cache = load_window_cache()" in reload_source
    assert "reloaded_window_counts == window_counts" in reload_source
    assert "raw = reloaded_raw" in reload_source


def test_window_feature_materialization_is_parallel_and_vectorizes_labels():
    source = _notebook_source()

    assert '"BVR_WINDOW_BUILD_WORKERS"' in source
    assert "executor.map(write_prepared_shard, prepared_episode_shards)" in source
    assert "episode_window_layout[episode_id]" in source
    assert "selected_targets == label" in source
    assert "np.fromiter(" not in source


def test_window_count_pass_reuses_compact_selection_metadata():
    source = _notebook_source()

    assert "episode_window_selection[episode_id]" in source
    assert "prepared_episodes.append(" in source
    assert 'feature_columns = ["episode_id", "time_s", *MODEL_FEATURE_COLUMNS]' in source
    assert "iter_dataset_episodes(shard_dataset, selection_columns)" in source
    assert "starts, mixed, encoded_targets = episode_window_selection[episode_id]" in source
    assert "del episode_window_selection" in source


def test_window_preparation_retains_features_and_avoids_a_second_parquet_scan():
    source = _notebook_source()

    assert 'feature_columns = ["episode_id", "time_s", *MODEL_FEATURE_COLUMNS]' in source
    assert "def prepare_window_shard(shard):" in source
    assert 'selection_columns = [*feature_columns, "tactical_label"]' in source
    assert "iter_dataset_episodes(shard_dataset, selection_columns)" in source
    assert "prepared_episodes.append(" in source
    assert "episode, starts, is_mixed, encoded_targets" in source
    assert "def write_prepared_shard(prepared_episodes):" in source
    assert "prepared_episode_shards = [prepared[3] for prepared in prepared_shards]" in source
    assert "executor.map(write_prepared_shard, prepared_episode_shards)" in source
    assert "def write_window_shard(shard):" not in source
    assert 'episode_skills["episode_id"].to_numpy(copy=False)' in source
    assert "episode_window_layout.keys() != episode_indices.keys()" in source


def test_window_materialization_uses_ram_and_contiguous_serialization():
    source = _notebook_source()

    assert "destination[\"features\"][sample_left:sample_right] = np.column_stack(" in source
    assert "for feature_index, column in enumerate(MODEL_FEATURE_COLUMNS)" not in source
    assert "np.empty(shape, dtype=dtype)" in source
    assert "np.save(path, array, allow_pickle=False)" in source
    assert "serialization_workers = min(4, WINDOW_BUILD_WORKERS, len(cache_arrays))" in source
    assert "executor.map(save_cache_array, cache_arrays)" in source
    assert 'mmap_mode="r+"' in source
    assert 'array.flush()' not in source


def test_window_materialization_uses_additional_parallelism_for_the_hot_path():
    source = _notebook_source()

    assert '"BVR_WINDOW_BUILD_WORKERS"' in source
    assert "executor.map(write_prepared_shard, prepared_episode_shards)" in source
    assert 'split_sample_counts[split_name]' in source


def test_window_cache_reports_major_stage_timings_and_slowest_stage():
    source = _notebook_source()

    assert 'window_cell_started = time.perf_counter()' in source
    assert 'window_stage_seconds["Parquet scan/window selection"]' in source
    assert 'window_stage_seconds["in-memory cache writes"]' in source
    assert 'window_stage_seconds["contiguous cache serialization"]' in source
    assert '"[windows] Performance summary (wall time):"' in source
    assert 'f"[windows] Slowest measured stage: {slowest_stage}' in source


def test_normalized_window_cache_is_not_normalized_twice():
    source = _notebook_source()

    assert 'if window_cache_manifest.get("normalized", False):' in source
    assert '"normalization_mean": normalization_mean.tolist()' in source
    assert '"normalization_scale": normalization_scale.tolist()' in source
    assert 'print("Reused normalized trajectory maps and cached scaler parameters")' in source


def test_normalization_parallelizes_splits_without_increasing_chunk_budget():
    source = _notebook_source()

    assert '"BVR_PREPROCESS_WORKERS"' in source
    assert (
        "normalization_workers = min(PREPROCESS_WORKERS, len(raw), "
        "PREPROCESS_CHUNK_WINDOWS)"
    ) in source
    assert (
        "normalization_chunk_rows = max(1, PREPROCESS_CHUNK_WINDOWS // "
        "normalization_workers)"
    ) in source
    assert "ThreadPoolExecutor(max_workers=normalization_workers)" in source
    assert "executor.map(normalize_split, raw.values())" in source
    assert "chunk -= normalization_mean" in source
    assert "chunk /= normalization_scale" in source


def test_canonical_window_scan_avoids_per_row_string_decoding_and_estimates_speedup():
    source = _notebook_source()

    assert 'FAST_CANONICAL_SCAN = os.getenv("BVR_FAST_CANONICAL_SCAN", "1") == "1"' in source
    assert 'def assign_canonical_episodes_to_shards():' in source
    assert "scan_columns = list(MODEL_FEATURE_COLUMNS)" in source
    assert 'def iter_canonical_shard_episodes(shard, summaries):' in source
    assert 'columns=scan_columns, use_threads=False' in source
    assert 'summary["switch_time_s"] / SAMPLE_DT_S - 1e-12' in source
    assert 'projection_speedup_ceiling = old_projection_bytes / max(1, active_projection_bytes)' in source
    assert '24556.06 / projection_speedup_ceiling' in source


def test_compact_metadata_retains_schedule_for_fast_window_scan():
    source = _notebook_source()

    assert '"switch_time_s": float(schedule["switch_time_s"])' in source
    assert '"primary_label": schedule["primary"]' in source
    assert '"secondary_label": schedule["secondary"]' in source


def test_canonical_scan_converts_each_numeric_column_once_per_shard():
    source = _notebook_source()

    assert "table = parquet_file.read(columns=scan_columns, use_threads=False)" in source
    assert "arrays = {" in source
    assert "array[left:right]" in source
    assert "for summary in summaries:" in source
    assert "pieces = {column: [] for column in scan_columns}" not in source
    assert "conversion_call_reduction = canonical_episode_count / len(shards)" in source


def test_canonical_window_selection_uses_numeric_schedule_directly():
    source = _notebook_source()

    assert "def canonical_window_selection(sample_count, summary, target_dtype):" in source
    assert 'summary["switch_time_s"] / SAMPLE_DT_S - 1e-12' in source
    assert "mixed = (starts < switch_index) & (ends >= switch_index)" in source
    assert 'label_to_index[summary["primary_label"]]' in source
    assert 'label_to_index[summary["secondary_label"]]' in source
    assert 'episode["tactical_label"] = np.where(' not in source


def test_canonical_scan_reconstructs_regular_timestamps_without_reading_them():
    source = _notebook_source()

    assert "scan_columns = list(MODEL_FEATURE_COLUMNS)" in source
    assert "def canonical_window_selection(sample_count, summary, target_dtype):" in source
    assert 'summary["switch_time_s"] / SAMPLE_DT_S - 1e-12' in source
    assert "canonical_times = starts * SAMPLE_DT_S" in source
    assert "canonical_end_times = (starts + WINDOW_SAMPLES - 1) * SAMPLE_DT_S" in source
    scan_assignment = source.index("scan_columns = list(MODEL_FEATURE_COLUMNS)")
    scan_read = source.index("table = parquet_file.read(columns=scan_columns", scan_assignment)
    assert '"time_s"' not in source[scan_assignment:scan_read]
