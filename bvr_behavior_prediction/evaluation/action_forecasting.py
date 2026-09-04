def action_metrics(true_actions, predicted_actions, branches=("heading", "altitude", "speed", "fire")):
    total = len(true_actions)
    if not total: raise ValueError("At least one action is required")
    metrics = {}
    for i, name in enumerate(branches):
        metrics[f"{name}_accuracy"] = sum(t[i] == p[i] for t, p in zip(true_actions, predicted_actions)) / total
        metrics[f"{name}_bin_mae"] = sum(abs(t[i] - p[i]) for t, p in zip(true_actions, predicted_actions)) / total
    metrics["joint_accuracy"] = sum(tuple(t) == tuple(p) for t, p in zip(true_actions, predicted_actions)) / total
    return metrics

