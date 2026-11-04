"""Prepare, audit, train and evaluate exact future catalogue selections."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "bvr_sim_source"))

from bvr_behavior_prediction.forecasting.data import read_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "baseline", "train", "report", "status", "verify"))
    parser.add_argument("--corpus", type=Path, default=ROOT / "shared/corpora/trained-pilot-episodes-v1")
    parser.add_argument("--reference-splits", type=Path,
                        default=ROOT / "artifacts/motion_forecasting/data_v1/scenario_splits.json")
    parser.add_argument("--data", type=Path, default=ROOT / "artifacts/skill_forecasting/data_v1")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/skill_forecasting/initial_v1")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/training/skill_forecasting.json")
    parser.add_argument("--architecture", choices=("gru", "transformer"), default="gru")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = read_json(args.config)
    if args.command == "prepare":
        from bvr_behavior_prediction.skill_forecasting.data import prepare_dataset
        result = prepare_dataset(args.corpus, args.data, args.reference_splits, **config["data"])
        print(json.dumps(result["counts"], indent=2))
    elif args.command == "baseline":
        from bvr_behavior_prediction.skill_forecasting.evaluation import evaluate_baseline
        result = evaluate_baseline(args.data, args.output / "training_frequency")
        print(result.query("pilot_id == 'population' and cohort == 'all'").to_string(index=False))
    elif args.command == "train":
        from bvr_behavior_prediction.skill_forecasting.train import train_model
        train_model(ROOT, args.data, args.output / args.architecture, args.architecture,
                    config["training"], resume=args.resume)
    elif args.command == "verify":
        from bvr_behavior_prediction.skill_forecasting.train import verify_run
        print(json.dumps(verify_run(args.data, args.output / args.architecture), indent=2))
    elif args.command == "report":
        from bvr_behavior_prediction.skill_forecasting.evaluation import build_report
        print(build_report(args.output, args.data).to_string(index=False))
    else:
        for path in sorted(args.output.glob("*/progress.json")):
            print(path.parent.name, json.dumps(read_json(path), indent=2))


if __name__ == "__main__":
    main()
