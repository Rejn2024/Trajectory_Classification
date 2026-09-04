def expected_calibration_error(labels, confidence, predictions, bins=10):
    total = len(labels); error = 0.0
    for index in range(bins):
        low, high = index / bins, (index + 1) / bins
        members = [i for i, p in enumerate(confidence) if low <= p < high or (index == bins-1 and p == high)]
        if members:
            accuracy = sum(predictions[i] == labels[i] for i in members) / len(members)
            mean_confidence = sum(confidence[i] for i in members) / len(members)
            error += len(members) / total * abs(accuracy - mean_confidence)
    return error

