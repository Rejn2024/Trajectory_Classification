"""Validate and publish the curated 20-replay archive without raw duplicates."""

import hashlib
import json
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    source = ROOT / "artifacts/tactical_checkpoint_replays"
    archive_path = source.with_suffix(".zip")
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    checksums = json.loads((source / "checksums.json").read_text(encoding="utf-8"))
    with zipfile.ZipFile(archive_path) as archive:
        if archive.testzip() is not None:
            raise ValueError("Replay archive failed CRC validation")
        acmi = [name for name in archive.namelist() if name.endswith(".txt.acmi")]
        if len(acmi) != 20 or len(manifest["recordings"]) != 20:
            raise ValueError("Expected exactly 20 selected-checkpoint replays")
        for relative, expected in checksums.items():
            # Existing archive contains its top-level folder for easy extraction.
            candidates = [n for n in archive.namelist()
                          if n == relative or n.endswith("/" + relative)]
            if len(candidates) != 1:
                raise ValueError(f"Missing or ambiguous archive member: {relative}")
            if hashlib.sha256(archive.read(candidates[0])).hexdigest() != expected:
                raise ValueError(f"Replay checksum mismatch: {relative}")
        for row in manifest["recordings"]:
            if row["acmi_sha256"] != checksums[row["acmi_file"]]:
                raise ValueError("ACMI manifest mismatch")
    output = ROOT / "shared/replays/tactical-checkpoints-v1"
    output.mkdir(parents=True, exist_ok=True)
    destination = output / archive_path.name
    shutil.copyfile(archive_path, destination)
    manifest["archive"] = {
        "file": destination.name, "bytes": destination.stat().st_size,
        "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
        "recordings": 20,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    rows = []
    for item in manifest["recordings"]:
        rows.append(f"| {item['pilot_id']} | {item['category']} | {item['seed']} | "
                    f"{item['duration_s']:.1f} | {item['outcome']} |")
    (output / "README.md").write_text(
        "# Tactical checkpoint replays\n\n"
        "Twenty Tacview recordings from the completed tactical population's validation-selected "
        "checkpoints: one clean win and one contested encounter per pilot. These are fresh "
        "inference recordings, not intermediate training-epoch demonstrations.\n\n"
        "[Download the compressed replay archive](tactical_checkpoint_replays.zip). "
        "Extract it and open any `replays/pilot_*/\u002a.txt.acmi` file in Tacview. "
        "The archive includes the encounter index, individual summaries and checksums.\n\n"
        "```powershell\ngit lfs pull --include=\"shared/replays/tactical-checkpoints-v1/**\" --exclude=\"\"\n"
        "Expand-Archive shared/replays/tactical-checkpoints-v1/tactical_checkpoint_replays.zip "
        "-DestinationPath artifacts/tacview\n```\n\n"
        "The learned observer is **A01**, shown as **Red** by the native simulator. "
        "The scripted tactical opponent is **B01**, shown as **Blue**. Both are F-16s. "
        "The RL adapter's blue/red terminology is reversed, so use these aircraft IDs.\n\n"
        "These encounters were selected for visual interest from an earlier assessment. "
        "They are separate from the 3,000 fresh forecasting episodes and do not form an "
        "unbiased performance sample. All 20 reproduced their archived outcome category; "
        "trajectories can differ slightly with inference hardware and batching.\n\n"
        "The ACMI 2.1 headers, timestamps, finite coordinates, aircraft identities, checksums "
        "and ZIP integrity have been checked. Visual playback in the Tacview GUI has not "
        "been inspected.\n\n"
        "| Pilot | Selection | Seed | Duration s | Outcome |\n"
        "| --- | --- | ---: | ---: | --- |\n" + "\n".join(rows) + "\n\n"
        "[Manifest and checkpoint hashes](manifest.json)\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest["archive"], indent=2))


if __name__ == "__main__":
    main()
