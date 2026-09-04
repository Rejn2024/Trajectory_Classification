def classification_metrics(y_true, probabilities):
    import numpy as np
    from sklearn.metrics import balanced_accuracy_score, f1_score, log_loss
    probs = np.asarray(probabilities); predictions = probs.argmax(1)
    one_hot = np.eye(probs.shape[1])[np.asarray(y_true)]
    return {"macro_f1": f1_score(y_true, predictions, average="macro"),
            "balanced_accuracy": balanced_accuracy_score(y_true, predictions),
            "nll": log_loss(y_true, probs), "brier": float(np.mean(np.sum((probs-one_hot)**2, axis=1)))}

