"""Run a bounded training-only search and a frozen validation comparison."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bvr_behavior_prediction.skill_forecasting.improvement import (
    confirm,
    evaluate,
    fit_run,
    initialize,
    refit,
    screen,
    summarize_search,
    verify_and_report,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("initialize", "screen", "confirm", "refit", "evaluate", "status", "fit", "verify"))
    parser.add_argument("--data", type=Path, default=ROOT / "artifacts/skill_forecasting/data_v1")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/skill_forecasting/improvement_v2")
    parser.add_argument("--candidate", default="control")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--design", type=Path)
    args = parser.parse_args()
    if args.command == "initialize":
        result = initialize(ROOT, args.data, args.output, design_path=args.design)
    elif args.command == "screen":
        result = screen(ROOT, args.output)
    elif args.command == "confirm":
        result = confirm(ROOT, args.output)
    elif args.command == "refit":
        result = refit(ROOT, args.output)
    elif args.command == "evaluate":
        result = evaluate(ROOT, args.output)
    elif args.command == "fit":
        result = fit_run(ROOT, args.output, args.candidate, fold=args.fold, seed=args.seed)
    elif args.command == "verify":
        result = verify_and_report(ROOT, args.output)
    else:
        print(summarize_search(args.output).to_string(index=False))
        return
    if result is not None:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
