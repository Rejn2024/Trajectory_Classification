def transition_metrics(y_true, probability):
    from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
    return {"auroc": roc_auc_score(y_true, probability),
            "pr_auc": average_precision_score(y_true, probability),
            "brier": brier_score_loss(y_true, probability),
            "nll": log_loss(y_true, [[1-p, p] for p in probability])}

