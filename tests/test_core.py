import math
import unittest

from bvr_behavior_prediction.data.labels import annotate_transitions, kinematic_labels, Lateral
from bvr_behavior_prediction.data.observable_columns import MODEL_FEATURE_COLUMNS
from bvr_behavior_prediction.data.privileged_columns import PRIVILEGED_COLUMNS
from bvr_behavior_prediction.data.relational_features import relational_features
from bvr_behavior_prediction.data.transforms import to_observer_body, wrap_angle
from bvr_behavior_prediction.data.windows import episode_split
from bvr_behavior_prediction.evaluation.leakage_checks import assert_disjoint_splits
from bvr_behavior_prediction.simulator.aircraft_registry import get_aircraft
from bvr_behavior_prediction.simulator.bvr_adapter import BVRSimAdapter


class CoreTests(unittest.TestCase):
    def test_angle_wrap(self):
        self.assertAlmostEqual(wrap_angle(3 * math.pi), -math.pi)
        self.assertAlmostEqual(wrap_angle(-2 * math.pi), 0)

    def test_coordinate_rotation(self):
        x, y, z = to_observer_body((0, 10, 2), math.pi / 2)
        self.assertAlmostEqual(x, 10); self.assertAlmostEqual(y, 0, places=8); self.assertEqual(z, 2)

    def test_relative_geometry_and_invariance(self):
        observer = dict(x=0, y=0, z=1000, vx=100, vy=0, vz=0, heading=0, speed=100)
        target = dict(x=1000, y=0, z=1200, vx=50, vy=0, vz=0, heading=math.pi, speed=50)
        result = relational_features(observer, target)
        self.assertAlmostEqual(result["range"], math.sqrt(1_040_000))
        self.assertLess(result["range_rate"], 0)
        rotated_observer = {**observer, "vx": 0, "vy": 100, "heading": math.pi/2}
        rotated_target = {**target, "x": 0, "y": 1000, "vx": 0, "vy": 50, "heading": -math.pi/2}
        rotated = relational_features(rotated_observer, rotated_target)
        for key in ("rel_x_body", "rel_y_body", "range", "range_rate"):
            self.assertAlmostEqual(result[key], rotated[key], places=7)

    def test_native_action(self):
        self.assertEqual(BVRSimAdapter.validate_action((14, 14, 8, 1)), (14, 14, 8, 1))
        with self.assertRaises(ValueError): BVRSimAdapter.validate_action((15, 0, 0, 0))

    def test_labels_and_transitions(self):
        self.assertEqual(kinematic_labels(-0.2, 0, 0)[0], Lateral.TURN_LEFT)
        rows = [{"time_s": i * .4, "target_skill": "A" if i < 3 else "B"} for i in range(6)]
        annotate_transitions(rows)
        self.assertEqual(rows[0]["transition_within_2s"], 1)
        self.assertEqual(rows[3]["transition_flag"], 1)
        self.assertEqual(rows[-1]["target_next_skill"], "B")

    def test_no_leakage_and_episode_split(self):
        self.assertTrue(set(MODEL_FEATURE_COLUMNS).isdisjoint(PRIVILEGED_COLUMNS))
        split = episode_split([str(i) for i in range(20)])
        assert_disjoint_splits(split)
        self.assertEqual(sum(map(len, split.values())), 20)

    def test_registry(self): self.assertEqual(get_aircraft("f16").bvr_unit_spec, "F16")


if __name__ == "__main__": unittest.main()

