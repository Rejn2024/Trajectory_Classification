"""One inference owner batches histories from independent simulator processes.

CUDA is created only in this owner. The frozen policy weights remain independent;
vmap evaluates all ten policies in one ensemble forward, without averaging them.
"""

from __future__ import annotations

import copy
import time
import traceback
from multiprocessing.connection import wait

# Keep Torch first for Windows DLL initialization, including standalone imports.
# isort: off
import torch
import numpy as np
# isort: on
from torch import nn
from torch.func import functional_call, stack_module_state, vmap


class PolicyHeads(nn.Module):
    """The frozen actor's exact encoder/heads, omitting unused critic/log-probs."""

    def __init__(self, pilot):
        super().__init__()
        self.pilot = pilot

    def forward(self, histories):
        p = self.pilot
        tokens = p.input_projection(histories) + p.position_embedding[:, -histories.shape[1] :]
        hidden = p.encoder_norm(p.encoder(tokens)[:, -1])
        return p.skill_head(hidden), p.parameter_mean(hidden)


class BatchedActors:
    def __init__(self, models, device="cuda"):
        if not models:
            raise ValueError("Provide frozen pilots")
        self.names = sorted(models)
        self.models = models
        self.device = torch.device(device)
        # Transformer fastpath has no portable vmap rule. SDPA/linear layers do.
        torch.backends.mha.set_fastpath_enabled(False)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        wrappers = [PolicyHeads(models[name]).eval() for name in self.names]
        reference = wrappers[0].pilot
        for model in wrappers[1:]:
            if (
                model.pilot.skill_names != reference.skill_names
                or model.pilot.parameter_names != reference.parameter_names
                or model.pilot.history_steps != reference.history_steps
                or model.pilot.skill_conditioned_parameters
                != reference.skill_conditioned_parameters
            ):
                raise ValueError("Ensemble requires identical actor contracts")
        with torch.inference_mode():
            params, buffers = stack_module_state(wrappers)
            self.params = {k: v.detach().to(self.device) for k, v in params.items()}
            self.buffers = {k: v.detach().to(self.device) for k, v in buffers.items()}
        self.base = copy.deepcopy(wrappers[0]).to("meta")
        self.forward = vmap(lambda p, b, x: functional_call(self.base, (p, b), (x,)))
        self.indices = {name: i for i, name in enumerate(self.names)}

    def heads(self, requests):
        """Return per-request logits/means; padding is never returned as real data."""
        if not requests:
            return [], {"rows": 0, "padded_rows": 0}
        positions, counts = [], [0] * len(self.names)
        first = np.asarray(requests[0][1], dtype=np.float32)
        for name, history in requests:
            i = self.indices[name]
            if np.asarray(history).shape != first.shape:
                raise ValueError("Incompatible policy history shapes")
            positions.append((i, counts[i]))
            counts[i] += 1
        slots = max(counts)
        batch = np.zeros((len(self.names), slots, *first.shape), dtype=np.float32)
        for (_, history), (i, j) in zip(requests, positions, strict=True):
            batch[i, j] = history
        with torch.inference_mode():
            logits, means = self.forward(
                self.params, self.buffers, torch.as_tensor(batch, device=self.device)
            )
            # One combined device-to-host transfer, then decode in request order.
            logits, means = logits.cpu().numpy(), means.cpu().numpy()
        return [(logits[i, j], means[i, j]) for i, j in positions], {
            "rows": len(requests),
            "padded_rows": len(self.names) * slots,
        }

    def act(self, requests):
        heads, stats = self.heads(requests)
        actions = []
        for (name, _), (logits, means) in zip(requests, heads, strict=True):
            model = self.models[name]
            index = int(np.argmax(logits))
            if model.skill_conditioned_parameters:
                means = means.reshape(len(model.skill_names), len(model.parameter_names))[index]
            actions.append((model.skill_names[index], model.decode_parameters(index, means)))
        return actions, stats

    def parity_check(self, histories=8):
        """Compare every frozen actor against CPU reference outputs before rollouts."""
        rng = np.random.default_rng(731)
        requests = [
            (name, rng.normal(0, 0.3, (self.models[name].history_steps,
                                     self.models[name].input_projection.in_features)).astype(np.float32))
            for name in self.names
            for _ in range(histories)
        ]
        actual, stats = self.heads(requests)
        max_error, min_margin = 0.0, float("inf")
        for name in self.names:
            subset = [(r, a) for r, a in zip(requests, actual, strict=True) if r[0] == name]
            x = torch.as_tensor(np.stack([r[0][1] for r in subset]), dtype=torch.float32)
            with torch.inference_mode():
                logits, means = PolicyHeads(self.models[name])(x)
            for (_, (got_logits, got_means)), ref_logits, ref_means in zip(
                subset, logits.numpy(), means.numpy(), strict=True
            ):
                if not (
                    np.allclose(got_logits, ref_logits, atol=2e-5, rtol=2e-5)
                    and np.allclose(got_means, ref_means, atol=2e-5, rtol=2e-5)
                    and got_logits.argmax() == ref_logits.argmax()
                ):
                    raise ValueError("Batched actor failed the CPU reference parity check")
                max_error = max(
                    max_error,
                    float(np.max(np.abs(got_logits - ref_logits))),
                    float(np.max(np.abs(got_means - ref_means))),
                )
                ordered = np.sort(ref_logits)
                min_margin = min(min_margin, float(ordered[-1] - ordered[-2]))
        return {
            "histories": len(requests),
            "max_absolute_error": max_error,
            "minimum_argmax_margin": min_margin,
            "passed": True,
            **stats,
        }


class GroupedActors:
    """Batch independent actors by architecture, preserving request order."""

    def __init__(self, models, device="cuda"):
        groups = {}
        for name, model in models.items():
            signature = (tuple((k, tuple(v.shape)) for k, v in model.state_dict().items()),
                         tuple(layer.self_attn.num_heads for layer in model.encoder.layers),
                         model.history_steps, model.skill_conditioned_parameters)
            groups.setdefault(signature, {})[name] = model
        self.groups = [BatchedActors(group, device) for group in groups.values()]

    def act(self, requests):
        actions = [None] * len(requests)
        stats = {"rows": 0, "padded_rows": 0}
        for group in self.groups:
            selected = [(i, request) for i, request in enumerate(requests)
                        if request[0] in group.indices]
            if not selected:
                continue
            results, counts = group.act([request for _, request in selected])
            for (i, _), action in zip(selected, results, strict=True):
                actions[i] = action
            for key in stats:
                stats[key] += counts[key]
        if any(action is None for action in actions):
            raise ValueError("Unknown pilot in inference request")
        return actions, stats

    def parity_check(self, histories=8):
        results = [group.parity_check(histories) for group in self.groups]
        return {"passed": all(r["passed"] for r in results), "architecture_groups": len(results),
                "groups": results}


def serve(work, connections, stop, ready, device, batch_wait_ms, max_requests):
    """Spawn target. Requests contain both aircraft; replies preserve their order."""
    from pathlib import Path

    from generate_nn_duel_corpus import STATE, initialize_worker, read, write

    work = Path(work)
    stats = {
        "device": device,
        "dispatches": 0,
        "rows": 0,
        "padded_rows": 0,
        "inference_seconds": 0.0,
        "max_rows_per_dispatch": 0,
    }
    pending = []
    try:
        initialize_worker(work)
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        config = read(work / "config.json")
        models = {
            name: STATE["loader"](work / "tactical_snapshot/training" / name, device="cpu")
            for name in sorted(config["pilots"])
        }
        actors = (GroupedActors if config.get("population_kind") == "varied_transformers_v1"
                  else BatchedActors)(models, device)
        stats["parity"] = actors.parity_check()
        stats["gpu_name"] = torch.cuda.get_device_name(0) if device == "cuda" else None
        ready.send({"ready": True, **stats})
        active = list(connections)
        while not stop.is_set():
            available = wait(active, timeout=0.1)
            if not available:
                continue
            pending = []
            deadline = time.perf_counter() + batch_wait_ms / 1000
            while available:
                for connection in available:
                    try:
                        request = connection.recv()
                    except EOFError:
                        active.remove(connection)
                        continue
                    pending.append((connection, request))
                if len(pending) >= max_requests or time.perf_counter() >= deadline:
                    break
                remaining = [c for c in active if c not in [p[0] for p in pending]]
                available = (
                    wait(remaining, timeout=max(0.0, deadline - time.perf_counter()))
                    if remaining
                    else []
                )
            if not pending:
                continue
            requests = [row for _, request in pending for row in request]
            start = time.perf_counter()
            actions, dispatch = actors.act(requests)
            stats["inference_seconds"] += time.perf_counter() - start
            stats["dispatches"] += 1
            stats["rows"] += dispatch["rows"]
            stats["padded_rows"] += dispatch["padded_rows"]
            stats["max_rows_per_dispatch"] = max(stats["max_rows_per_dispatch"], dispatch["rows"])
            offset = 0
            for connection, request in pending:
                connection.send({"actions": actions[offset : offset + len(request)]})
                offset += len(request)
            pending = []
    except BaseException as error:
        failure = {"error": repr(error), "traceback": traceback.format_exc()}
        try:
            ready.send(failure)
        except (BrokenPipeError, OSError):
            pass
        for connection, _ in pending:
            try:
                connection.send(failure)
            except (BrokenPipeError, OSError):
                pass
        stats.update(failure)
        raise
    finally:
        stats["mean_rows_per_dispatch"] = stats["rows"] / max(1, stats["dispatches"])
        write(work / "inference_metrics.json", stats)
        ready.close()
        for c in connections:
            c.close()
