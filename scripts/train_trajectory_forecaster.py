"""Prepare and train future speed, heading and altitude regressors (GRU/Transformer)."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    from bvr_behavior_prediction.forecasting.data import prepare_dataset, read_json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "train", "report", "status"))
    parser.add_argument("--corpus", type=Path, required=False)
    parser.add_argument("--data", type=Path, default=ROOT / "artifacts/trajectory_forecasting/data_v1")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/trajectory_forecasting/runs_v1")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/training/trajectory_forecasting.json")
    parser.add_argument("--architecture", choices=("gru", "transformer"), default="transformer")
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = read_json(args.config)
    if args.command == "prepare":
        if args.corpus is None:
            parser.error("prepare requires --corpus pointing to newly collected pilot episodes")
        result = prepare_dataset(args.corpus, args.data, target="flight_state", **config["data"])
        print(json.dumps(result["counts"], indent=2))
    elif args.command == "train":
        from bvr_behavior_prediction.forecasting.flight_state import CONTRACT
        from bvr_behavior_prediction.forecasting.train import train_model

        if read_json(args.data / "data_manifest.json").get("target_contract") != CONTRACT:
            parser.error("prepare flight-state regression targets before training")
        if args.device:
            config["training"]["device"] = args.device
        train_model(ROOT, args.data, args.output / args.architecture, args.architecture,
                    config["training"], resume=args.resume)
    elif args.command == "report":
        from bvr_behavior_prediction.forecasting.evaluation import build_report
        from bvr_behavior_prediction.forecasting.flight_state import CONTRACT

        paths = sorted(args.output.glob("*/validation_metrics.csv"))
        if not paths:
            parser.error("No validation metrics found")
        for path in paths:
            if read_json(path.parent / "run.json").get("target_contract") != CONTRACT:
                parser.error("Output contains a different forecasting target contract")
        print(build_report(args.output).to_string())
    else:
        for path in sorted(args.output.glob("*/progress.json")):
            print(path.parent.name, json.dumps(read_json(path), indent=2))


if __name__ == "__main__":
    main()
