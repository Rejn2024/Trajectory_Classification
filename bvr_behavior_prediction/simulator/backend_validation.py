def rollout_signature(adapter, actions, seed: int = 0) -> list[str]:
    """Return repr-based signatures useful for deterministic/backend smoke comparisons."""
    observation, _ = adapter.reset(seed)
    signature = [repr(observation)]
    for action in actions:
        observation, _, terminated, truncated, _ = adapter.step(action)
        signature.append(repr(observation))
        if terminated or truncated:
            break
    return signature

