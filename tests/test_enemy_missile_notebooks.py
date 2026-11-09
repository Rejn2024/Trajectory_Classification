"""Notebook contracts stay aligned across the three controlled architecture runs."""

import ast
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[1]
NOTEBOOKS = {
    "transformer": "18_enemy_missile_nn_duel_skill_forecasting.ipynb",
    "missile_set": "19_enemy_missile_set_nn_duel_skill_forecasting.ipynb",
    "typed_graph": "20_enemy_missile_graph_nn_duel_skill_forecasting.ipynb",
}
COMPARISON = "21_compare_enemy_missile_nn_duel_skill_forecasting.ipynb"


def load(name):
    return json.loads((ROOT / "notebooks" / name).read_text())


def code(cell):
    source = "".join(cell["source"])
    assert source.startswith("%%time\n")
    return source.split("\n", 1)[1]


@pytest.mark.parametrize("name", [*NOTEBOOKS.values(), COMPARISON])
def test_notebooks_have_portable_schema_and_compile_without_running_training(name):
    notebook = load(name)
    assert notebook["nbformat"] == 4
    ids = [c["id"] for c in notebook["cells"]]
    assert len(ids) == len(set(ids))
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is None and cell["outputs"] == []
            compile(code(cell), f"{name}:cell{index}", "exec")


def recipe(architecture):
    setup = ast.parse(code(load(NOTEBOOKS[architecture])["cells"][1]))
    selected = []
    for statement in setup.body:
        if isinstance(statement, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id in {"ARCHITECTURE", "CONFIG", "TRAINING_SEEDS"}
            for target in statement.targets
        ):
            selected.append(statement)
        if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call):
            function = statement.value.func
            if isinstance(function, ast.Attribute) and isinstance(function.value, ast.Name):
                if function.value.id == "CONFIG" and function.attr == "update":
                    selected.append(statement)
    namespace = {"DEFAULT_TRAINING": {}, "DEVICE": "cpu"}
    exec(compile(ast.Module(body=selected, type_ignores=[]), "recipe", "exec"), namespace)
    return namespace


def test_training_recipes_differ_only_in_architecture():
    configurations = []
    for architecture in NOTEBOOKS:
        namespace = recipe(architecture)
        assert namespace["ARCHITECTURE"] == architecture
        assert namespace["TRAINING_SEEDS"] == [42, 43, 44]
        config = namespace["CONFIG"].copy()
        assert config.pop("architecture") == architecture
        assert config["width"] == 96 and config["heads"] == 6 and config["layers"] == 2
        assert config["dropout"] == .25
        assert config["l1_lambda"] == 1e-6
        assert config["class_weighting"] == "sqrt_inverse"
        assert config["max_class_weight_ratio"] == 5
        configurations.append(config)
    assert configurations[0] == configurations[1] == configurations[2]


@pytest.mark.parametrize("architecture", NOTEBOOKS)
def test_saved_diagnostics_cover_all_runs_before_the_receipt_is_published(architecture):
    notebook = load(NOTEBOOKS[architecture])
    setup = code(notebook["cells"][1])
    training = code(notebook["cells"][3])
    final = code(notebook["cells"][5])
    assert "EnemyMissileNNDuelData(DATA)" in setup
    assert "REEVALUATE_SAVED = False" in setup
    assert 'saved_run["data_manifest_sha256"]' in training
    tree = ast.parse(final)
    export_loops = [
        statement for statement in tree.body if isinstance(statement, ast.For)
        and any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "save_diagnostics" for node in ast.walk(statement))
    ]
    assert len(export_loops) == 1
    assert ast.unparse(export_loops[0].iter) == "RUNS.items()"
    export_line = export_loops[0].lineno
    receipt_lines = [node.lineno for node in ast.walk(tree)
                     if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                     and node.func.id == "write_json"]
    assert receipt_lines and all(line > export_line for line in receipt_lines)
    assert '"data_manifest_sha256": sha256(DATA / "manifest.json")' in final


def test_comparison_reads_saved_runs_without_training_or_preparing_data():
    notebook = load(COMPARISON)
    sources = "\n".join(code(c) for c in notebook["cells"] if c["cell_type"] == "code")
    tree = ast.parse(sources)
    calls = {node.func.id for node in ast.walk(tree)
             if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
    assert "compare_architectures" in calls
    assert not calls & {"train_model", "final_evaluation", "prepare_dataset", "EnemyMissileNNDuelData"}
    assert 'EXPLICIT_RUNS = {}' in sources
    assert '"transformer", "missile_set", "typed_graph"' in sources


@pytest.mark.parametrize("resume,changed_loss,accepted", [
    (False, False, True), (True, False, False), (False, True, False),
])
def test_historical_baseline_configuration_is_report_only_compatible(
        tmp_path, resume, changed_loss, accepted):
    import hashlib
    from types import SimpleNamespace

    import pandas as pd

    config = recipe("transformer")["CONFIG"]
    requested = {**config, "seed": 42}
    historical = {k: v for k, v in requested.items()
                  if k not in {"architecture", "graph_width", "graph_layers"}}
    if changed_loss:
        historical["l1_lambda"] *= 2
    data_path = tmp_path / "data"
    data_path.mkdir()
    (data_path / "manifest.json").write_text("prepared dataset identity")
    digest = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
    run = tmp_path / "seed42"
    run.mkdir()
    pd.DataFrame({"epoch": [1], "test_nll": [1.]}).to_csv(run / "history.csv", index=False)
    (run / "training_status.json").write_text(json.dumps({"status": "completed"}))
    (run / "run.json").write_text(json.dumps({
        "training": historical, "data_manifest_sha256": digest(data_path / "manifest.json"),
        "classes": ["a", "b"], "feature_names": ["f1", "f2"],
    }))
    namespace = {
        "pd": pd, "RUNS": {42: run}, "CONFIG": config, "TRACKING": {},
        "TRAIN_MODEL": False, "RESUME": resume, "REEVALUATE_SAVED": False,
        "SPLITS": ("train", "test", "validation"), "DEVICE": "cpu", "DATA": data_path,
        "data": SimpleNamespace(classes=["a", "b"], manifest={"features": ["f1", "f2"]}),
        "read_json": lambda path: json.loads(Path(path).read_text()), "sha256": digest,
        "display": lambda *args: None,
    }
    executable = code(load(NOTEBOOKS["transformer"])["cells"][3])
    if accepted:
        exec(compile(executable, "report-only-baseline", "exec"), namespace)
        assert len(namespace["histories"][42]) == 1
    else:
        with pytest.raises(ValueError, match="Requested configuration differs"):
            exec(compile(executable, "report-only-baseline", "exec"), namespace)


@pytest.mark.parametrize("architecture", tuple(NOTEBOOKS))
@pytest.mark.parametrize("complete_finals,reevaluate,accepted", [
    (True, False, True), (False, False, False), (True, True, False),
])
def test_historical_evaluation_settings_only_read_complete_saved_outputs(
        tmp_path, architecture, complete_finals, reevaluate, accepted):
    """A newly introduced evaluation policy must not silently replace old scores."""
    config = {**recipe(architecture)["CONFIG"], "seed": 42}
    historical = {k: v for k, v in config.items()
                  if k not in {"evaluation_precision", "evaluation_attention"}}
    if complete_finals:
        for name in ("final_metrics.csv", "final_per_class.csv", "selection.json",
                     "final_train.npz", "final_test.npz", "final_validation.npz"):
            (tmp_path / name).touch()
    source = code(load(NOTEBOOKS[architecture])["cells"][3])
    start = source.index('    saved_training = dict(saved_run["training"])')
    end = source.index('    display(histories[seed].tail())', start)
    guard = "\n".join(line[4:] for line in source[start:end].splitlines())
    namespace = {
        "TRAIN_MODEL": False, "RESUME": False, "REEVALUATE_SAVED": reevaluate,
        "SPLITS": ("train", "test", "validation"), "run": tmp_path, "seed": 42,
        "config": config, "saved_run": {"training": historical},
    }
    if accepted:
        exec(compile(guard, "historical-evaluation-policy", "exec"), namespace)
    else:
        with pytest.raises(ValueError):
            exec(compile(guard, "historical-evaluation-policy", "exec"), namespace)


def test_comparison_notebook_audits_its_input_run_receipt(tmp_path):
    import hashlib

    import pandas as pd

    output = tmp_path / "comparison"
    output.mkdir()

    def compare(runs, destination, bootstrap_resamples):
        (destination / "manifest.json").write_text(json.dumps({"artifacts": {}}))
        return {"metrics": pd.DataFrame({"nll": [1.]})}

    digest = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
    namespace = {
        "RUNS": {architecture: [tmp_path / architecture] for architecture in NOTEBOOKS},
        "OUTPUT": output, "BOOTSTRAP_RESAMPLES": 0, "RECEIPTS": {},
        "compare_architectures": compare, "pd": pd, "display": lambda *args: None,
        "read_json": lambda path: json.loads(Path(path).read_text()),
        "write_json": lambda path, data: Path(path).write_text(json.dumps(data)), "sha256": digest,
    }
    executable = code(load(COMPARISON)["cells"][3])
    exec(compile(executable, "comparison-report", "exec"), namespace)
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["artifacts"]["input_run_lists.json"] == digest(output / "input_run_lists.json")
