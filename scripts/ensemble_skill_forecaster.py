"""Confirm fixed mixtures on training groups, refit, evaluate once and verify."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bvr_behavior_prediction.skill_forecasting.ensemble_search import (
    confirm_mixtures,
    evaluate_mixture,
    refit_mixture,
    register,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("register", "confirm", "refit", "evaluate", "verify"))
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/skill_forecasting/improvement_v3")
    args = parser.parse_args()
    if args.command == "register":
        result = register(ROOT, args.output)
    elif args.command == "confirm":
        result = confirm_mixtures(ROOT, args.output)
    elif args.command == "refit":
        result = refit_mixture(ROOT, args.output)
    else:
        result = evaluate_mixture(ROOT, args.output, verify=args.command == "verify")
    if result is not None:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
