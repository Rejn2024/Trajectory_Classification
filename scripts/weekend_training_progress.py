"""Small JSON progress records and notebook tqdm views; no simulator imports."""

import json
import statistics
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


def timing_windows(path, size=10):
    """Separate growing flight/update work from slower execution of the same work.

    Old campaign rows already have rollout/minibatch counts. Missing newer
    validation and memory measurements stay absent rather than becoming zeros.
    """
    if size < 1:
        raise ValueError("Timing window size must be positive")
    rows = metric_rows(Path(path))
    result = []
    fields = {
        "epoch_s": "epoch_seconds",
        "rollout_s": "rollout_seconds",
        "PPO_s": "update_seconds",
        "validation_s": "evaluation_seconds",
        "decisions": "rollout_transitions",
        "minibatches": "update_minibatches",
        "PPO_passes": "update_passes_completed",
        "validation_decisions": "evaluation_transitions",
        "KL": "approx_kl",
        "CUDA_peak_MB": "cuda_peak_allocated_mb",
    }
    for left in range(0, len(rows), size):
        window = rows[left:left + size]
        summary = {"first_epoch": window[0]["epoch"], "last_epoch": window[-1]["epoch"]}
        for label, key in fields.items():
            values = [row[key] for row in window if key in row]
            if values:
                summary[label] = statistics.median(values)
        for label, seconds, count in (
            ("rollout_ms/decision", "rollout_seconds", "rollout_transitions"),
            ("PPO_ms/minibatch", "update_seconds", "update_minibatches"),
            ("validation_ms/decision", "evaluation_seconds", "evaluation_transitions"),
        ):
            values = [1000 * row[seconds] / row[count] for row in window
                      if seconds in row and row.get(count, 0) > 0]
            if values:
                summary[label] = statistics.median(values)
        result.append(summary)
    return result


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


class TextProgress:
    """Append readable progress events when notebook widgets are unavailable."""

    def __init__(self, *, total, desc, unit, initial=0, **options):
        self.total, self.description, self.unit = total, desc, unit
        self.n, self.postfix = initial, ""
        self.started = time.monotonic()

    def update(self, delta):
        self.n += delta

    def set_postfix_str(self, value, refresh=False):
        self.postfix = value

    def refresh(self):
        elapsed = duration(time.monotonic() - self.started)
        print(f"{self.description}: {self.n}/{self.total} {self.unit} "
              f"[elapsed {elapsed}, {self.postfix}]", flush=True)

    def close(self):
        pass


class CampaignProgress:
    """Display current-pilot, overall-training and collection bars in Jupyter."""

    def __init__(self, job, flights, bar_factory=None, clock=time.time):
        if bar_factory is None:
            from tqdm.auto import tqdm

            widget = any(cls.__module__ == "tqdm.notebook" for cls in tqdm.__mro__)
            bar_factory = tqdm if widget else TextProgress
        self.job, self.flights, self.bar_factory, self.clock = (
            Path(job),
            flights,
            bar_factory,
            clock,
        )
        self.overall = self.pilot = self.collection = None
        self.pilot_key = None
        self.last_state = None
        self.bar_states = {}
        self.widget = any(
            cls.__module__ == "tqdm.notebook" for cls in getattr(bar_factory, "__mro__", ())
        )

    def new_bar(self, description, total, unit, leave=True, initial=0):
        return self.bar_factory(
            total=total,
            initial=initial,
            desc=description,
            unit=unit,
            leave=leave,
            dynamic_ncols=True,
            # refresh() below owns rendering; update() must not draw an extra
            # line with the previous postfix before we explicitly refresh it.
            mininterval=float("inf"),
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} {unit} "
            "[elapsed {elapsed}, {rate_fmt}{postfix}]",
        )

    def update_bar(self, bar, count, postfix, state):
        key = (count, state)
        if not self.widget and self.bar_states.get(id(bar)) == key:
            return
        bar.set_postfix_str(postfix, refresh=False)
        bar.update(max(0, count - bar.n))
        bar.refresh()
        self.bar_states[id(bar)] = key

    def refresh(self):
        stage = read_json(self.job / "status.json").get("stage", "starting")
        record = read_json(self.job / "training_progress.json")
        collection = read_json(self.job / "collection/progress.json") if stage in (
            "corpus", "completed"
        ) else {}
        state = (stage, record.get("pilot"), record.get("phase"),
                 record.get("overall_completed_epochs"), collection.get("completed"))
        # Forced terminal redraws every two seconds become permanent notebook
        # output when ipywidgets is unavailable (megabytes in the saved run).
        # Widgets still refresh every poll; text only renders changed progress.
        if not self.widget and state == self.last_state:
            return
        self.last_state = state
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
                stage,
            )
            if active:
                key = record["stage"], record["pilot"]
                if key != self.pilot_key:
                    if self.pilot is not None:
                        self.bar_states.pop(id(self.pilot), None)
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
                    (record["phase"], record.get("loss"), record.get("validation_return")),
                )
            elif self.pilot is not None:
                self.bar_states.pop(id(self.pilot), None)
                self.pilot.close()
                self.pilot, self.pilot_key = None, None
        if stage in ("corpus", "completed"):
            record = collection
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
                self.update_bar(self.collection, count, f"{stage}, collection ETA≈{eta}", stage)

    def close(self):
        self.last_state = None
        self.refresh()
        for bar in (self.pilot, self.overall, self.collection):
            if bar is not None:
                bar.close()
