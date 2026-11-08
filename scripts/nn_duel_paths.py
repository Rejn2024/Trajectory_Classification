"""Check snapshot paths before extraction on Windows with traditional path limits."""

from __future__ import annotations

import os
import zipfile
from pathlib import PureWindowsPath


def check_snapshot_paths(archive, work_directories, *, windows=None):
    """Reject overlong archive destinations before creating a partial snapshot."""
    if windows is None:
        windows = os.name == "nt"
    if not windows:
        return None
    with zipfile.ZipFile(archive) as source:
        members = [info.filename for info in source.infolist() if not info.is_dir()]
    destinations = (
        str(PureWindowsPath(work) / "tactical_snapshot" / member)
        for work in work_directories
        for member in members
    )
    # Win32 counts UTF-16 code units, including two for non-BMP characters.
    def length(value):
        return len(value.encode("utf-16-le")) // 2

    longest = max(destinations, key=length, default="")
    maximum = length(longest)
    if maximum >= 260:
        raise RuntimeError(
            f"Windows snapshot path would be {maximum} characters (maximum 259): {longest}. "
            r"Set BVR_NN_WORK to a short parent such as C:\bvrw, then restart the kernel "
            "and Run All. BVR_NN_OUTPUT can remain in the repository. "
            "No simulator snapshot has been extracted by this check."
        )
    return maximum
