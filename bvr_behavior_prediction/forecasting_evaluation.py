"""Bounded frame reuse and device-resident epoch metrics for skill forecasting.

Caches belong to a single frozen evaluation pass. Final evaluation can use the
same frame iterator while retaining every prediction through the full evaluator.
"""

import math

import numpy as np
import pandas as pd
import torch

from . import paired_skill_forecasting as p


def _frame_failure(tensor, stage, rows, steps, ids, data, split, batch, precision, attention):
    """Map a failing unique frame back to every affected history/observer."""
    bad = (~torch.isfinite(tensor)).flatten(1).any(1).nonzero().flatten().cpu().numpy()
    affected = np.isin(steps, rows[bad])
    selected = np.flatnonzero(affected.any(1))
    first_bad_frame = steps[selected, affected[selected].argmax(1)]
    tensor_rows = np.searchsorted(rows, first_bad_frame)
    try:
        p._check_evaluation_tensor(tensor[torch.as_tensor(tensor_rows, device=tensor.device)], stage,
                                   split, batch, ids[selected], data, precision, attention)
    except p.NonfiniteEvaluationError as error:
        error.details.update(frame_indices=rows[bad[:16]].tolist(), frame_cache=True)
        raise


def evaluation_batches(model, data, split, device, batch_size, precision, pipeline,
                         attention, frame_cache, frame_batch_size, cache_max_frames):
    """Yield aligned IDs, inputs, targets, and whether inputs are frame encodings."""
    ids = data.indices[split]
    if not frame_cache:
        batches = (pipeline.batches(ids, batch_size) if pipeline is not None else
                   (data.batch(ids[start:start + batch_size], device)
                    for start in range(0, len(ids), batch_size)))
        offset = 0
        for x, y in batches:
            yield ids[offset:offset + len(x)], x, y, False
            offset += len(x)
        return
    if (not hasattr(model, "encode_frame_rows") or not hasattr(data, "features")
            or not hasattr(data, "anchors")):
        raise ValueError("Frame caching requires a frame-local missile model and prepared feature rows/anchors")
    if (isinstance(frame_batch_size, bool) or not isinstance(frame_batch_size, int) or frame_batch_size < 1
            or isinstance(cache_max_frames, bool) or not isinstance(cache_max_frames, int)
            or cache_max_frames < p.HISTORY_STEPS):
        raise ValueError("Frame cache needs positive frame_batch_size and cache_max_frames >= 51")
    # Even disjoint/unsorted histories fit the cap; overlap makes the actual cache smaller.
    windows_per_chunk = cache_max_frames // p.HISTORY_STEPS
    if windows_per_chunk >= batch_size:
        windows_per_chunk = windows_per_chunk // batch_size * batch_size
    device = torch.device(device)
    offsets = np.arange(1 - p.HISTORY_STEPS, 1)
    normalized_cache = getattr(pipeline, "cache", None)
    for start in range(0, len(ids), windows_per_chunk):
        chunk_ids = ids[start:start + windows_per_chunk]
        steps = np.asarray(data.anchors[chunk_ids, None]) + offsets
        if steps.min() < 0 or steps.max() >= len(data.features):
            raise ValueError("Evaluation history references feature rows outside the prepared array")
        unique, inverse = np.unique(steps, return_inverse=True)
        inverse = inverse.reshape(steps.shape)

        def frame_batches():
            for f in range(0, len(unique), frame_batch_size):
                rows = unique[f:f + frame_batch_size]
                if normalized_cache is not None:
                    yield (normalized_cache.features[torch.as_tensor(rows, device=device)],)
                else:
                    features = (data.features[rows] - data.mean) / data.std
                    host = torch.from_numpy(features)
                    yield (host.pin_memory() if device.type == "cuda" else host,)

        batches = frame_batches()
        if normalized_cache is None and device.type == "cuda":
            batches = p.prefetch_to_cuda(batches, device)
        encoded_rows = []
        frame_offset = 0
        for (frames,) in batches:
            rows = unique[frame_offset:frame_offset + len(frames)]
            def diagnose(tensor, stage):
                _frame_failure(tensor, stage, rows, steps, chunk_ids, data, split,
                                start // batch_size, precision, attention)
            p._check_evaluation_tensors([("inputs", frames)], split, start // batch_size,
                chunk_ids, data, precision, attention, diagnose=diagnose)
            with p.autocast_context(device, precision):
                encoded = model.encode_frame_rows(frames)
            p._check_evaluation_tensors([("frame_encoder_output", encoded)],
                split, start // batch_size, chunk_ids, data, precision, attention,
                diagnose=diagnose)
            encoded_rows.append(encoded)
            frame_offset += len(frames)
        encodings = torch.cat(encoded_rows)
        del encoded_rows
        for offset in range(0, len(chunk_ids), batch_size):
            sample_ids = chunk_ids[offset:offset + batch_size]
            index = torch.as_tensor(inverse[offset:offset + batch_size], device=device)
            targets = torch.as_tensor(data.targets[sample_ids], dtype=torch.long, device=device)
            yield sample_ids, encodings[index], targets, True


class _TaskTotals:
    def __init__(self, classes, device, allow_empty=False):
        self.classes, self.allow_empty = classes, allow_empty
        c = len(classes)
        self.confusion = torch.zeros((len(p.HORIZONS), c, c), device=device, dtype=torch.int64)
        self.loss_sum = torch.zeros((len(p.HORIZONS), c), device=device, dtype=torch.float64)

    def update(self, values):
        targets, probability, ce = (values[k] for k in ("targets", "probabilities", "losses"))
        valid = targets >= 0
        c = len(self.classes)
        true_index = torch.arange(len(p.HORIZONS), device=targets.device)[None] * c + targets.clamp_min(0)
        index = true_index * c + probability.argmax(-1)
        # Fixed-size scatter avoids the host synchronization of dynamic CUDA bincount.
        self.confusion.flatten().scatter_add_(0, index.flatten(), valid.long().flatten())
        self.loss_sum.flatten().scatter_add_(0, true_index.flatten(), (ce.double() * valid).flatten())

    def tables(self, split, confusion, sums):
        summary, scores = [], []
        c = len(self.classes)
        for h, horizon in enumerate(p.HORIZONS):
            cm = confusion[h]
            support, predicted, tp = cm.sum(1), cm.sum(0), cm.diagonal()
            count = int(support.sum())
            if not count and not self.allow_empty:
                raise ValueError(f"No labels for {split}, +{horizon}s")
            precision = np.divide(tp, predicted, out=np.zeros(c), where=predicted > 0)
            recall = np.divide(tp, support, out=np.zeros(c), where=support > 0)
            f1 = np.divide(2 * tp, support + predicted, out=np.zeros(c), where=support + predicted > 0)
            cnll = np.divide(sums[h], support, out=np.full(c, np.nan), where=support > 0)
            summary.append(dict(split=split, horizon_s=horizon, count=count,
                nll=float(sums[h].sum() / count) if count else np.nan,
                accuracy=float(tp.sum() / count) if count else np.nan, macro_f1=float(f1.mean()),
                supported_macro_f1=float(f1[support > 0].mean()) if count else np.nan,
                weighted_f1=float(np.dot(f1, support) / count) if count else np.nan))
            scores.extend(dict(split=split, horizon_s=horizon, skill=name, support=int(support[k]),
                class_nll=float(cnll[k]), precision=precision[k], recall=recall[k], f1=f1[k])
                for k, name in enumerate(self.classes))
        return {"summary": pd.DataFrame(summary), "per_class": pd.DataFrame(scores), "confusion": confusion}


def evaluate_streaming(model, data, split, device, batch_size, precision, pipeline, class_weights,
                       attention, intent_loss_weight, hierarchical_loss, family_reward, handedness_reward,
                       handedness_weights, frame_cache, frame_batch_size, cache_max_frames):
    """Exactly the epoch summaries/objective, without sample-sized prediction arrays."""
    multitask, hierarchical = getattr(model, "num_intents", 0) == 3, hasattr(model, "num_families")
    totals = {"skill": _TaskTotals(data.classes, device)}
    if multitask:
        totals["intent"] = _TaskTotals(data.intent_classes, device)
    mapping = None
    if hierarchical:
        totals.update(family=_TaskTotals(data.family_classes, device),
                      direction=_TaskTotals(data.direction_classes, device, allow_empty=True))
        mapping = {"family": model.skill_family_index,
                   "pair": torch.where(model.directional_skills, model.skill_pair_index, -1),
                   "direction": model.skill_direction_index}
    settings = p.hierarchical_loss_settings(dict(hierarchical_loss=hierarchical_loss,
                                                family_reward=family_reward, handedness_reward=handedness_reward))
    weights = None if class_weights is None else torch.as_tensor(class_weights, dtype=torch.float64, device=device)
    # joint weighted NLL, skill objective, family objective, family accuracy, hard/expected credit, credit loss.
    diagnostics = torch.zeros((7, len(p.HORIZONS)), dtype=torch.float64, device=device)
    log_a = math.log(family_reward)
    log_b = math.log(handedness_reward) if handedness_reward else -float("inf")
    denominator = float(np.logaddexp(log_a, log_b))
    a, b = math.exp(log_a - denominator), math.exp(log_b - denominator)
    batches = evaluation_batches(model, data, split, device, batch_size, precision, pipeline,
                                  attention, frame_cache, frame_batch_size, cache_max_frames)
    with p.evaluation_context(device, attention):
        for batch_index, (ids, x, y, encoded) in enumerate(batches):
            tasks = p._evaluation_batch(model, x, y, data, ids, device, precision, split, batch_index,
                                        attention, mapping, encoded)
            for name, values in tasks.items():
                totals[name].update(values)
            outer = 1. if weights is None else weights.gather(1, y.clamp_min(0).T).T
            diagnostics[0].add_((tasks["skill"]["losses"].double() * outer).sum(0))
            if hierarchical:
                family_ce, side_ce = tasks["family"]["losses"], tasks["direction"]["losses"]
                paired, valid = tasks["direction"]["targets"] >= 0, y >= 0
                hand = p.target_handedness_weights(y, mapping, handedness_weights)
                objective = p.hierarchical_example_loss(family_ce, side_ce, paired, handedness_weight=hand, **settings)
                diagnostics[1].add_((objective.double() * outer).sum(0))
                diagnostics[2].add_((family_ce.double() * outer).sum(0))
                family_correct = tasks["family"]["probabilities"].argmax(-1) == tasks["family"]["targets"]
                side_correct = tasks["direction"]["probabilities"].argmax(-1) == tasks["direction"]["targets"]
                credit = torch.where(paired, family_correct.double() * (a + b * side_correct.double()), family_correct.double())
                expected = torch.where(paired, a * (-family_ce).exp() + b * (-family_ce - side_ce).exp(),
                                        (-family_ce).exp())
                credit_loss = p.hierarchical_example_loss(family_ce, side_ce, paired, **{**settings,
                    "hierarchical_loss": "partial_credit"})
                diagnostics[3:].add_(torch.stack([v.double().mul(valid).sum(0) for v in
                    (family_correct, credit, expected, credit_loss)]))
    # Copy small aggregate buffers only after the split. No per-example probabilities leave the device.
    confusion = torch.cat([t.confusion.flatten() for t in totals.values()]).cpu().numpy()
    floats = torch.cat([t.loss_sum.flatten() for t in totals.values()] + [diagnostics.flatten()]).cpu().numpy()
    results, ci, fi = {}, 0, 0
    for name, total in totals.items():
        cn, fn = total.confusion.numel(), total.loss_sum.numel()
        results[name] = total.tables(split, confusion[ci:ci + cn].reshape(total.confusion.shape),
                                     floats[fi:fi + fn].reshape(total.loss_sum.shape))
        ci, fi = ci + cn, fi + fn
    diagnostic = floats[fi:].reshape(diagnostics.shape)
    counts = results["skill"]["summary"]["count"].to_numpy()
    mean = lambda values: float((values / counts).mean())
    result = results.pop("skill")
    result.update(results, data_loss=mean(diagnostic[0]), sample_indices=data.indices[split])
    if multitask:
        result["skill_data_loss"] = result["data_loss"]
        result["data_loss"] += intent_loss_weight * float(result["intent"]["summary"].nll.mean())
    if hierarchical:
        previous = result["skill_data_loss"]
        result.update(joint_skill_data_loss=previous, skill_data_loss=mean(diagnostic[1]),
                       family_data_loss=mean(diagnostic[2]))
        result["handedness_data_loss"] = result["skill_data_loss"] - result["family_data_loss"]
        result["data_loss"] += result["skill_data_loss"] - previous
        result["hierarchical_reward"] = pd.DataFrame([
            dict(split=split, horizon_s=h, count=int(counts[i]),
                 **{k: float(diagnostic[j + 3, i] / counts[i]) for j, k in enumerate(
                    ("family_accuracy", "hierarchical_credit", "expected_hierarchical_credit", "hierarchical_credit_loss"))})
            for i, h in enumerate(p.HORIZONS)])
    return result
