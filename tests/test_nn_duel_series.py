"""Every Run All starts a separate 20k series without reusing prior campaign data."""

import json
import secrets
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import generate_nn_duel_corpus as duel


@pytest.mark.parametrize("override_roots", [False, True])
def test_first_cell_restarts_at_zero_and_retains_previous_series(tmp_path, monkeypatch, override_roots):
    notebook = json.loads((ROOT / "notebooks/11_nn_vs_nn_training_data.ipynb").read_text())
    source = "".join(notebook["cells"][1]["source"]).removeprefix("%%time\n")
    startup = source[: source.index("# Test the native library mix")]
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts/generate_nn_duel_corpus.py").touch()
    monkeypatch.setenv("BVR_NN_ROOT", str(tmp_path))
    monkeypatch.syspath_prepend(str(tmp_path / "scripts"))
    for key in ("BVR_NN_WORK", "BVR_NN_OUTPUT", "BVR_NN_FLIGHTS", "BVR_NN_SEED"):
        monkeypatch.delenv(key, raising=False)
    old_work = tmp_path / "artifacts/nn_duels_batched_v1"
    old_output = tmp_path / "shared/corpora/nn-duels-batched-v1"
    old_work.mkdir(parents=True)
    old_output.mkdir(parents=True)
    (old_work / "config.json").write_text('{"flights": 10000, "seed": 26110600}')
    (old_output / "manifest.json").write_text('{"physical_flights": 1400}')
    if override_roots:
        monkeypatch.setenv("BVR_NN_WORK", str(old_work))
        monkeypatch.setenv("BVR_NN_OUTPUT", str(old_output))
    seeds = iter((100_000, 200_000))
    monkeypatch.setattr(secrets, "randbelow", lambda _: next(seeds))
    preserved = {p: p.read_bytes() for p in (old_work / "config.json", old_output / "manifest.json")}
    scope = {}
    exec(compile(startup, "notebook startup", "exec"), scope)  # noqa: S102
    first_work, first_output, first_seed = scope["WORK"], scope["OUTPUT"], scope["SEED"]
    assert scope["FLIGHTS"] == 20_000
    assert first_work.parent == (
        old_work if override_roots else tmp_path / "artifacts/nn_duels_20k_v1"
    )
    assert first_output.parent == (
        old_output if override_roots else tmp_path / "shared/corpora/nn-duels-20k-v1"
    )
    assert first_work.name == first_output.name == scope["SERIES_ID"]
    assert not list(first_work.iterdir()) and not first_output.exists()
    # Simulate a partly acquired series and cached benchmark/verification state.
    first_output.mkdir(parents=True)
    for path in (
        first_work / "config.json",
        first_work / "benchmark_batched.json",
        first_work / "verified_snapshot.json",
        first_work / "progress.json",
        first_output / "manifest.json",
    ):
        path.write_text('{"completed": 100}')
        preserved[path] = path.read_bytes()
    exec(compile(startup, "notebook startup", "exec"), scope)  # noqa: S102
    assert scope["WORK"] != first_work and scope["OUTPUT"] != first_output
    assert scope["SEED"] != first_seed
    assert scope["FLIGHTS"] == 20_000
    assert not list(scope["WORK"].iterdir()) and not scope["OUTPUT"].exists()
    assert all(path.read_bytes() == content for path, content in preserved.items())


def test_twenty_thousand_pairings_cover_each_ordered_pair_two_hundred_times():
    cases = duel.scenarios(20_000, seed=27_000_000)
    assert len(cases) == len({c["scenario_group"] for c in cases}) == 20_000
    assert len({json.dumps(c["initial"]) for c in cases}) == 20_000
    assert Counter(tuple(c["pilots"].values()) for c in cases) == {
        (f"pilot_{a:03d}", f"pilot_{b:03d}"): 200 for a in range(1, 11) for b in range(1, 11)
    }
    assert set(Counter(c["family"] for c in cases).values()) == {2000}


@pytest.mark.parametrize("corpus", ["nn-duels-20k-v1", "nn-duels-batched-v1"])
def test_fresh_series_preserves_hashed_bytes_with_windows_git_line_endings(
    tmp_path, monkeypatch, corpus
):
    git = shutil.which("git")
    if not git or not shutil.which("git-lfs"):
        pytest.skip("Git and Git LFS required for checkout integration")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(tmp_path / "global.gitconfig"))
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")

    def git_run(*args):
        return subprocess.check_output([git, *args], cwd=tmp_path, stderr=subprocess.PIPE)

    git_run("init")
    git_run("config", "--local", "core.autocrlf", "true")
    git_run("lfs", "install", "--local")
    shutil.copyfile(ROOT / ".gitattributes", tmp_path / ".gitattributes")
    directory = tmp_path / "shared/corpora" / corpus / "example_series"
    collector = directory / "collector/nn_duel_source.py"
    collector.parent.mkdir(parents=True)
    files = {
        collector: b"# Exact Windows source bytes\r\nVALUE = 1\r\n",
        directory / "manifest.json": b'{"physical_flights": 100}\n',
        directory / "config.json": b'{"flights": 20000}\n',
        directory / "README.md": b"# Fresh series\n",
        directory / "A01/observations/part-00000.parquet": b"PAR1\x00fake-payload\x00PAR1",
    }
    for path, content in files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    git_run("add", "--", ".gitattributes", directory.relative_to(tmp_path).as_posix())
    for path, content in files.items():
        staged = git_run("show", ":" + path.relative_to(tmp_path).as_posix())
        if path.suffix == ".parquet":
            assert staged.startswith(b"version https://git-lfs.github.com/spec/v1\n")
        else:
            assert staged == content
        path.unlink()
    git_run("checkout-index", "--all", "--force")
    assert all(path.read_bytes() == content for path, content in files.items())
