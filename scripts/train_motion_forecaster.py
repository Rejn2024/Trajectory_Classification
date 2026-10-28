"""Prepare raw pilot episodes and train position/velocity forecasters."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bvr_behavior_prediction.forecasting.data import prepare_dataset, read_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "train", "report", "status"))
    parser.add_argument("--corpus", type=Path, default=ROOT / "shared/corpora/trained-pilot-episodes-v1")
    parser.add_argument("--data", type=Path, default=ROOT / "artifacts/motion_forecasting/data_v1")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/motion_forecasting/initial_v1")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/training/motion_forecasting.json")
    parser.add_argument("--architecture", choices=("gru", "transformer"), default="transformer")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = read_json(args.config)
    if args.command == "prepare":
        result = prepare_dataset(args.corpus, args.data, **config["data"])
        print(json.dumps(result["counts"], indent=2))
    elif args.command == "train":
        from bvr_behavior_prediction.forecasting.train import train_model
        train_model(ROOT, args.data, args.output / args.architecture, args.architecture,
                    config["training"], resume=args.resume)
    elif args.command == "report":
        from bvr_behavior_prediction.forecasting.evaluation import build_report
        print(build_report(args.output).to_string())
    else:
        for path in sorted(args.output.glob("*/progress.json")):
            print(path.parent.name, json.dumps(read_json(path), indent=2))


if __name__ == "__main__":
    main()
