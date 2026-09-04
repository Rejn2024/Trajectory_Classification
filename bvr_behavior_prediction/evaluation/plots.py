def reliability_diagram(labels, probabilities, output_path, bins=10):
    import matplotlib.pyplot as plt
    points = []
    for index in range(bins):
        low, high = index / bins, (index + 1) / bins
        members = [i for i, p in enumerate(probabilities) if low <= p < high]
        if members:
            points.append((sum(probabilities[i] for i in members) / len(members),
                           sum(labels[i] for i in members) / len(members)))
    figure, axis = plt.subplots(); axis.plot([0, 1], [0, 1], "--", color="grey")
    if points: axis.plot(*zip(*points), marker="o")
    axis.set(xlabel="Mean confidence", ylabel="Observed frequency", xlim=(0, 1), ylim=(0, 1))
    figure.savefig(output_path, bbox_inches="tight"); plt.close(figure)

