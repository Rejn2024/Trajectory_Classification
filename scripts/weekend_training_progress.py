"""Small JSON progress records and notebook tqdm views; no simulator imports."""

import json
import time
from pathlib import Path


def read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def metric_rows(path):
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf8").splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            break  # A killed process can leave an unfinished final line.
    return rows


def duration(seconds):
    if seconds is None:
        return "estimating"
    seconds = max(0, int(seconds))
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    return f"{hours:d}:{minutes:02d}:{seconds:02d}"


class EpochProgressReporter:
    """Report only committed epochs; estimate future work from observed timings."""

    def __init__(self, job, settings, stage, pilot, write, clock=time.time):
        self.job, self.stage, self.pilot = Path(job), stage, pilot
        self.write, self.clock = write, clock
        self.epoch_started = None
        self.targets = {s: settings[f"{s}_epochs"] for s in ("simple", "tactical")}
        self.counts = dict.fromkeys(self.targets, 0)
        self.seconds = dict.fromkeys(self.targets, 0.0)
        self.pilot_count = 10
        for s in self.targets:
            for directory in (self.job / s).glob("pilot_*"):
                if s == stage and directory.name == pilot:
                    continue
                rows = metric_rows(directory / "training_metrics.jsonl")
                self.counts[s] += min(rows[-1]["epoch"], self.targets[s]) if rows else 0
                self.seconds[s] += sum(row["epoch_seconds"] for row in rows)

    def __call__(self, phase, epoch, metrics):
        stamp = self.clock()
        if phase == "collecting flights":
            self.epoch_started = stamp
        active = phase in (
            "collecting flights",
            "PPO update",
            "validation flights",
            "saving checkpoint",
        )
        counts, seconds = dict(self.counts), dict(self.seconds)
        counts[self.stage] += epoch
        seconds[self.stage] += sum(row["epoch_seconds"] for row in metrics)
        means = {s: seconds[s] / counts[s] if counts[s] else None for s in counts}
        fallback = means[self.stage] or next((v for v in means.values() if v is not None), None)
        pilot_mean = (
            sum(row["epoch_seconds"] for row in metrics[-5:]) / len(metrics[-5:])
            if metrics
            else fallback
        )
        pilot_eta = (self.targets[self.stage] - epoch) * pilot_mean if pilot_mean else None
        overall_eta = (
            sum(
                (self.pilot_count * self.targets[s] - counts[s]) * (means[s] or fallback)
                for s in counts
            )
            if fallback
            else None
        )
        self.write(
            self.job / "training_progress.json",
            {
                "stage": self.stage,
                "pilot": self.pilot,
                "phase": phase,
                "completed_epochs": epoch,
                "target_epochs": self.targets[self.stage],
                "overall_completed_epochs": sum(counts.values()),
                "overall_target_epochs": self.pilot_count * sum(self.targets.values()),
                "pilot_eta_seconds": pilot_eta,
                "training_eta_seconds": overall_eta,
                "epoch_started_at": self.epoch_started if active else None,
                "updated_at": stamp,
                "loss": metrics[-1]["loss"] if metrics else None,
                "validation_return": metrics[-1]["mean_return"] if metrics else None,
            },
        )


class CampaignProgress:
    """Display current-pilot, overall-training and collection bars in Jupyter."""

    def __init__(self, job, flights, bar_factory=None, clock=time.time):
        if bar_factory is None:
            from tqdm.auto import tqdm

            bar_factory = tqdm
        self.job, self.flights, self.bar_factory, self.clock = (
            Path(job),
            flights,
            bar_factory,
            clock,
        )
        self.overall = self.pilot = self.collection = None
        self.pilot_key = None

    def new_bar(self, description, total, unit, leave=True, initial=0):
        return self.bar_factory(
            total=total,
            initial=initial,
            desc=description,
            unit=unit,
            leave=leave,
            dynamic_ncols=True,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} {unit} "
            "[elapsed {elapsed}, {rate_fmt}{postfix}]",
        )

    @staticmethod
    def update_bar(bar, count, postfix):
        bar.update(max(0, count - bar.n))
        bar.set_postfix_str(postfix, refresh=False)
        bar.refresh()

    def refresh(self):
        stage = read_json(self.job / "status.json").get("stage", "starting")
        record = read_json(self.job / "training_progress.json")
        if record:
            if self.overall is None:
                self.overall = self.new_bar(
                    "Overall training",
                    record["overall_target_epochs"],
                    "epochs",
                    initial=record["overall_completed_epochs"],
                )
            active = stage in ("simple", "tactical") and stage == record["stage"]
            age = (
                max(0.0, self.clock() - record["epoch_started_at"])
                if active and record.get("epoch_started_at") is not None
                else 0.0
            )

            def eta(key):
                value = record.get(key)
                if value is not None and value > 0 and age >= value:
                    return "re-estimating"
                return duration(max(0.0, value - age) if value is not None else None)

            self.update_bar(
                self.overall,
                record["overall_completed_epochs"],
                f"{stage}, training ETA≈{eta('training_eta_seconds')}",
            )
            if active:
                key = record["stage"], record["pilot"]
                if key != self.pilot_key:
                    if self.pilot is not None:
                        self.pilot.close()
                    self.pilot = self.new_bar(
                        f"{key[0]} · {key[1]}",
                        record["target_epochs"],
                        "epochs",
                        leave=False,
                        initial=record["completed_epochs"],
                    )
                    self.pilot_key = key
                score = (
                    f", loss={record['loss']:.4f}, return={record['validation_return']:.2f}"
                    if record.get("loss") is not None
                    else ""
                )
                self.update_bar(
                    self.pilot,
                    record["completed_epochs"],
                    f"{record['phase']}, pilot ETA≈{eta('pilot_eta_seconds')}{score}",
                )
            elif self.pilot is not None:
                self.pilot.close()
                self.pilot, self.pilot_key = None, None
        if stage in ("corpus", "completed"):
            record = read_json(self.job / "collection/progress.json")
            if record:
                if self.collection is None:
                    self.collection = self.new_bar(
                        "NN-v-NN collection",
                        self.flights,
                        "flights",
                        initial=record.get("completed", 0),
                    )
                count, rate = record.get("completed", 0), record.get("flights_per_hour", 0.0)
                eta = duration((self.flights - count) * 3600 / rate if rate else None)
                self.update_bar(self.collection, count, f"{stage}, collection ETA≈{eta}")

    def close(self):
        self.refresh()
        for bar in (self.pilot, self.overall, self.collection):
            if bar is not None:
                bar.close()
