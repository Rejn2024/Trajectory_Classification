def stratify(rows, metric, key="target_aircraft_type"):
    groups = {}
    for row in rows: groups.setdefault(row[key], []).append(row)
    return {name: metric(group) for name, group in groups.items()}

