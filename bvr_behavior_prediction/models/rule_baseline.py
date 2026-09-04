from ..data.labels import kinematic_labels


class RuleBaseline:
    def predict_one(self, row):
        return kinematic_labels(row["target_turn_rate"], row["target_climb_rate"],
                                row["target_acceleration"])
    def predict(self, rows): return [self.predict_one(row) for row in rows]


def persistence_forecast(current_labels): return list(current_labels)
def repeat_action_forecast(recent_actions): return [tuple(actions[-1]) for actions in recent_actions]

