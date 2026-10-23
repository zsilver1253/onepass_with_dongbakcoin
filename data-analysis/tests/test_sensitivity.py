"""sensitivity·step3_sensitivity 테스트. 실행: data-analysis 폴더에서 python -m unittest discover tests"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sensitivity import load_inputs, one_at_a_time, scenario_grid  # noqa: E402
from step3_sensitivity import per_user_credit_won, total_credit_eok  # noqa: E402

INPUTS = pd.DataFrame({
    "key": ["x", "y"], "role": ["input", "input"],
    "low": [1.0, 1.0], "mid": [2.0, 2.0], "high": [3.0, 10.0], "unit": ["", ""], "status": ["가정", "가정"],
}).set_index("key")


def product(v: dict) -> float:
    return v["x"] * v["y"]


class SensitivityTest(unittest.TestCase):
    def test_one_at_a_time_sorted_by_swing(self) -> None:
        t = one_at_a_time(INPUTS, product, ["x", "y"])
        self.assertEqual(t["key"].tolist(), ["y", "x"])  # y 범위가 더 넓어 위로
        self.assertEqual(t.loc[0, ["result_low", "result_mid", "result_high"]].tolist(), [2.0, 4.0, 20.0])
        self.assertEqual(t.loc[0, "ratio"], 10.0)

    def test_scenario_grid_shape_and_corner(self) -> None:
        g = scenario_grid(INPUTS, product, "x", "y")
        self.assertEqual(g.shape, (3, 3))
        self.assertEqual(g.loc[3.0, 10.0], 30.0)

    def test_load_inputs_rejects_reversed_range(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "inputs.csv"
            INPUTS.reset_index().assign(low=[5.0, 1.0]).to_csv(path, index=False, encoding="utf-8-sig")
            with self.assertRaises(ValueError):
                load_inputs(path)


class Step3ModelTest(unittest.TestCase):
    def test_units(self) -> None:
        v = {"a_spend_total": 10, "benefit_rate": 1.0, "user_share": 10, "pop_15plus": 1_000_000}
        self.assertAlmostEqual(total_credit_eok(v), 100.0)  # 10조 × 1% × 10% = 100억
        self.assertAlmostEqual(per_user_credit_won(v), 100_000.0)  # 10조 ÷ 100만 명 × 1% = 10만 원


if __name__ == "__main__":
    unittest.main()
