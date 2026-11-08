"""Dependency diagnostics that can run before importing forecasting modules."""

import importlib
import importlib.metadata
import sys


def require_pyarrow():
    """Check the actual Parquet extension and explain native import failures.

    A Windows policy rejection needs an approved binary or a policy-owner review;
    retrying imports or changing the notebook's dataset cannot resolve it.
    """
    try:
        return importlib.import_module("pyarrow.parquet")
    except (ImportError, OSError) as error:
        try:
            version = importlib.metadata.version("pyarrow")
        except importlib.metadata.PackageNotFoundError:
            version = "not installed"
        details = (
            f"Notebook Python: {sys.executable}\n"
            f"Environment prefix: {sys.prefix}\n"
            f"Installed PyArrow version: {version}\n"
            f"Original error: {type(error).__name__}: {error}\n"
        )
        message = str(error).lower()
        if "application control" in message and "block" in message:
            reason = (
                "Windows Application Control blocked a PyArrow native component or one "
                "of its dependencies. The exception does not identify the blocked file.\n"
                "Check Event Viewer > Applications and Services Logs > Microsoft > Windows "
                "> CodeIntegrity > Operational immediately after reproducing the failure. "
                "Ask the machine's policy administrator to review the named binary and "
                "provide a policy-approved PyArrow build or authorize that component. "
                "A notebook edit or switching CUDA/CPU cannot remove this policy block.\n"
            )
        elif isinstance(error, ModuleNotFoundError) and error.name == "pyarrow":
            reason = (
                "PyArrow is missing from the notebook kernel's environment. Install the "
                "project dependencies in the Python environment printed below, using "
                "your approved package source.\n"
            )
        else:
            reason = (
                "PyArrow's Parquet runtime could not load. Repair its installation and "
                "native dependencies in the notebook kernel's environment. This message "
                "alone does not establish an Application Control block.\n"
            )
        raise RuntimeError(
            reason + details +
            "After repairing the environment, restart the Jupyter kernel and retry. "
            "See docs/nn_duel_skill_forecasting.md (Windows PyArrow import failure)."
        ) from error
