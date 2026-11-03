"""Screen sequence models, confirm against v3, refit and evaluate one frozen recipe."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bvr_behavior_prediction.skill_forecasting.sequence_search import (
    confirm,
    evaluate,
    initialize,
    refit,
    screen,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("initialize", "screen", "confirm", "refit", "evaluate"))
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/skill_forecasting/improvement_v4")
    parser.add_argument("--data", type=Path, default=ROOT / "artifacts/skill_forecasting/data_v1")
    parser.add_argument("--baseline", type=Path, default=ROOT / "artifacts/skill_forecasting/improvement_v3")
    parser.add_argument("--design", type=Path, default=ROOT / "configs/training/skill_forecasting_sequence_search.json")
    args = parser.parse_args()
    if args.command == "initialize":
        result = initialize(ROOT, args.data, args.output, args.design, args.baseline)
    else:
        result = {"screen": screen, "confirm": confirm, "refit": refit, "evaluate": evaluate}[args.command](ROOT, args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
