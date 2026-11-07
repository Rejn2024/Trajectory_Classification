"""Discovered Git must remain available to publication subprocesses."""

import ast
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from bvr_behavior_prediction import paired_skill_forecasting as paired

ROOT = Path(__file__).parents[1]


def notebook_sources():
    notebook = json.loads((ROOT / "notebooks/11_nn_vs_nn_training_data.ipynb").read_text())
    first = "".join(notebook["cells"][1]["source"]).removeprefix("%%time\n")
    setup = "".join(notebook["cells"][3]["source"]).removeprefix("%%time\n")
    return first, setup[: setup.index("setup_sources =")]


def namespace(tmp_path):
    first, _ = notebook_sources()
    scope = {
        "ROOT": tmp_path,
        "WORK": tmp_path / "work",
        "OUTPUT": tmp_path / "output",
        "FLIGHTS": 10000,
        "SEED": 42,
        "WORKERS": 12,
        "EPISODE_SECONDS": 300,
        "SHARD_FLIGHTS": 100,
        "BACKEND": "local_cpu",
        "CUDA_AVAILABLE": False,
        "DETERMINISTIC": True,
        "BATCH_WAIT_MS": 2,
        "BATCH_MAX_REQUESTS": 32,
        "GIT_EXECUTABLE": None,
        "SCRIPT": tmp_path / "collector.py",
        "PUBLISH": True,
        "DOWNLOAD_LFS": False,
        "Path": Path,
        "os": os,
        "sys": sys,
        "subprocess": subprocess,
    }
    recorded = []
    scope["updates"] = SimpleNamespace(run=lambda args, **kwargs: recorded.append((args, kwargs)))
    function = next(
        n for n in ast.parse(first).body if isinstance(n, ast.FunctionDef) and n.name == "run"
    )
    exec(compile(ast.Module(body=[function], type_ignores=[]), "notebook run", "exec"), scope)  # noqa: S102
    return scope, recorded


def bootstrap(scope):
    _, source = notebook_sources()
    exec(compile(source, "notebook Git setup", "exec"), scope)  # noqa: S102 -- actual repository cell


@pytest.fixture
def windows_git(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT))
    git = tmp_path / "Program Files/Git/cmd/git.exe"
    git.parent.mkdir(parents=True)
    git.touch()
    state = SimpleNamespace(branch="forecasting_ed_261104", staged="")
    calls = []
    monkeypatch.setattr(paired.shutil, "which", lambda _: None)
    monkeypatch.setattr(
        paired,
        "os",
        SimpleNamespace(
            name="nt",
            environ={"ProgramFiles": str(tmp_path / "Program Files")},
            pathsep=os.pathsep,
        ),
    )

    def run(args, **kwargs):
        calls.append((args, kwargs))
        stdout = state.branch if args[1:3] == ["branch", "--show-current"] else state.staged
        return SimpleNamespace(returncode=0, stdout=stdout, stderr="")

    monkeypatch.setattr(paired.subprocess, "run", run)
    return git, state, calls


@pytest.mark.parametrize("download,publish", [(True, False), (False, True), (True, True)])
def test_windows_discovery_is_retained_for_child_path(tmp_path, windows_git, download, publish):
    git, _, calls = windows_git
    scope, recorded = namespace(tmp_path)
    scope.update(DOWNLOAD_LFS=download, PUBLISH=publish)
    bootstrap(scope)
    assert scope["GIT_EXECUTABLE"] == str(git)
    assert calls and all(args[0] == str(git) for args, _ in calls)
    assert any("pull" in args for args, _ in calls) == download
    scope["run"]("publish", backend="local_cpu")
    assert recorded[0][1]["env"]["PATH"].split(os.pathsep)[0] == str(git.parent)


def test_download_returns_the_resolved_executable(tmp_path, windows_git):
    git, _, _ = windows_git
    assert paired.download_paired_corpora(tmp_path, ["example.zip"]) == str(git)


def test_local_only_setup_does_not_require_git(tmp_path, monkeypatch):
    def unexpected_discovery(*args):
        raise AssertionError("Local-only setup should not look for Git")

    monkeypatch.setattr(paired, "resolve_git_runtime", unexpected_discovery)
    scope, _ = namespace(tmp_path)
    scope.update(DOWNLOAD_LFS=False, PUBLISH=False)
    bootstrap(scope)
    assert scope["GIT_EXECUTABLE"] is None


def test_publication_checks_git_even_when_download_is_disabled(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT))
    monkeypatch.setattr(paired.shutil, "which", lambda _: None)
    monkeypatch.setattr(paired, "os", SimpleNamespace(name="posix"))
    scope, recorded = namespace(tmp_path)
    with pytest.raises(RuntimeError, match="Git executable not found"):
        bootstrap(scope)
    assert recorded == []


@pytest.mark.parametrize("problem", ["branch", "staged"])
def test_publication_preflight_checks_repository_before_collection(tmp_path, windows_git, problem):
    _, state, _ = windows_git
    scope, recorded = namespace(tmp_path)
    setattr(state, problem, "unrelated-work")
    with pytest.raises(RuntimeError, match="Check out" if problem == "branch" else "unstage"):
        bootstrap(scope)
    assert recorded == []


def test_publication_child_finds_git_with_original_path_missing_git(tmp_path, monkeypatch):
    git = shutil.which("git")
    if not git or not shutil.which("git-lfs"):
        pytest.skip("Git and Git LFS required for child-process integration")
    monkeypatch.syspath_prepend(str(ROOT))
    empty_path = tmp_path / "empty_path"
    empty_path.mkdir()
    # Keep LFS available while deliberately removing Git from PATH.
    shutil.copy2(
        shutil.which("git-lfs"), empty_path / ("git-lfs.exe" if os.name == "nt" else "git-lfs")
    )
    monkeypatch.setenv("PATH", str(empty_path))
    assert shutil.which("git") is None
    real_resolve = paired.resolve_git_runtime
    monkeypatch.setattr(
        paired,
        "resolve_git_runtime",
        lambda root, git_executable=None: real_resolve(root, git_executable or git),
    )
    scope, _ = namespace(tmp_path)
    assert scope["GIT_EXECUTABLE"] is None
    scope["subprocess"] = SimpleNamespace(
        run=lambda args, **kwargs: SimpleNamespace(
            stdout="forecasting_ed_261104" if "branch" in args else "",
            returncode=0,
            stderr="",
        )
    )
    bootstrap(scope)
    child = tmp_path / "publication probe.py"
    child.write_text(
        "import subprocess\nsubprocess.run(['git', '--version'], check=True)\n",
        encoding="utf-8",
    )
    scope["SCRIPT"] = child
    executed = []

    def execute(args, **kwargs):
        result = subprocess.run(
            args, cwd=kwargs["cwd"], env=kwargs["env"], text=True, capture_output=True, check=True
        )
        executed.append(result.stdout)

    scope["updates"] = SimpleNamespace(run=execute)
    scope["run"]("publish", backend="local_cpu")
    assert executed and "git version" in executed[0]


@pytest.mark.parametrize("publish", [True, False])
def test_publication_lock_setting_reaches_child_without_changing_other_repositories(
    tmp_path, monkeypatch, publish
):
    git = shutil.which("git")
    if not git or not shutil.which("git-lfs"):
        pytest.skip("Git and Git LFS required for child-process integration")
    endpoint = "https://github.com/Rejn2024/Trajectory_Classification.git/info/lfs"
    key = f"lfs.{endpoint}.locksverify"
    unrelated = "lfs.https://github.com/example/locked-assets.git/info/lfs.locksverify"
    global_config = tmp_path / "global.gitconfig"
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(global_config))
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    monkeypatch.syspath_prepend(str(ROOT))

    def git_run(*args):
        return subprocess.run(
            [git, *args], cwd=tmp_path, text=True, capture_output=True, check=True
        ).stdout.strip()

    git_run("init", "-b", "forecasting_ed_261104")
    git_run("config", "--global", "lfs.locksverify", "true")
    git_run("config", "--global", key, "true")
    git_run("config", "--local", unrelated, "true")
    original_global = global_config.read_bytes()
    original_local = (tmp_path / ".git/config").read_bytes()
    hook = tmp_path / ".git/hooks/pre-push"
    hook.write_text("#!/bin/sh\ngit lfs pre-push \"$@\"\n", encoding="utf-8")
    original_hook = hook.read_bytes()
    scope, _ = namespace(tmp_path)
    scope["PUBLISH"] = publish
    bootstrap(scope)
    assert git_run("config", "--bool", "--get", key) == ("false" if publish else "true")
    assert git_run("config", "--bool", "--get", unrelated) == "true"
    assert git_run("config", "--bool", "--get", "lfs.locksverify") == "true"
    assert global_config.read_bytes() == original_global
    assert hook.read_bytes() == original_hook
    if not publish:
        assert (tmp_path / ".git/config").read_bytes() == original_local
        return

    # Exercise the notebook's actual subprocess path, including inherited Git config.
    child = tmp_path / "publication lock probe.py"
    child.write_text(
        "import subprocess\n"
        f"value = subprocess.check_output(['git', 'config', '--bool', '--get', {key!r}], "
        "text=True).strip()\n"
        "assert value == 'false', value\n",
        encoding="utf-8",
    )
    scope["SCRIPT"] = child

    def execute(args, **kwargs):
        subprocess.run(args, cwd=kwargs["cwd"], env=kwargs["env"], check=True)

    scope["updates"] = SimpleNamespace(run=execute)
    scope["run"]("publish", backend="local_cpu")
