import argparse
import json
from pathlib import Path

from ..data.schema import EPISODE_COLUMNS


class DatasetBuilder:
    def __init__(self, output_dir, manifest, shard_size=100):
        self.root = Path(output_dir); self.manifest = manifest; self.shard_size = shard_size

    def write(self, trajectory_rows: list[dict], episode_rows: list[dict]) -> None:
        """Write the canonical Parquet layout (imports storage dependencies only on use)."""
        import pandas as pd
        import yaml
        self.root.mkdir(parents=True, exist_ok=True); (self.root / "trajectories").mkdir(exist_ok=True)
        with (self.root / "manifest.yaml").open("w") as fh:
            yaml.safe_dump(self.manifest.as_dict(), fh, sort_keys=False)
        missing = set(EPISODE_COLUMNS) - set(episode_rows[0]) if episode_rows else set()
        if missing: raise ValueError(f"Episode metadata missing columns: {sorted(missing)}")
        pd.DataFrame(episode_rows).to_parquet(self.root / "episodes.parquet", index=False)
        episode_ids = list(dict.fromkeys(row["episode_id"] for row in trajectory_rows))
        for number, start in enumerate(range(0, len(episode_ids), self.shard_size)):
            ids = set(episode_ids[start:start + self.shard_size])
            pd.DataFrame([r for r in trajectory_rows if r["episode_id"] in ids]).to_parquet(
                self.root / "trajectories" / f"shard_{number:03d}.parquet", index=False)
        label_map = {"lateral": ["STRAIGHT", "TURN_LEFT", "TURN_RIGHT"],
                     "vertical": ["LEVEL", "CLIMB", "DESCEND"],
                     "energy": ["STEADY_SPEED", "ACCELERATE", "DECELERATE"],
                     "tactical": ["MAINTAIN", "PURSUE", "BEAM", "CRANK_LEFT", "CRANK_RIGHT", "EXTEND"]}
        (self.root / "label_map.json").write_text(json.dumps(label_map, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Build canonical BVR trajectory Parquet datasets")
    parser.add_argument("--output", required=True); parser.add_argument("--shard-size", type=int, default=100)
    parser.parse_args()
    parser.error("Use DatasetBuilder from a configured BVR Sim generation application")

