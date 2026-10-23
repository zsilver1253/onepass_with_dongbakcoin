"""industry_evidence 테스트. 실행: data-analysis 폴더에서 python -m unittest discover tests"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from industry_evidence import card_amounts, report_amounts, share_table  # noqa: E402

MAPPING = pd.DataFrame({"report_category": ["슈퍼", "식품", "주유"], "d1_category": ["유통", "음식료품", "불명"]})


def stats(rows: list[tuple]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["table_id", "period", "category", "value"])


def basis(monthly: bool) -> dict:
    return {"table_id": "T", "total_category": "총계", "value_is_monthly_avg": monthly,
            "periods": {"P": ["2025-06", "2025-08"]}}


class ReportAmountsTest(unittest.TestCase):
    def test_monthly_avg_times_months_and_drops_total_and_unknown(self) -> None:
        s = stats([("T", "P", "슈퍼", 2.0), ("T", "P", "주유", 9.0), ("T", "P", "총계", 11.0)])
        out = report_amounts(s, MAPPING, basis(monthly=True))
        self.assertEqual(out["d1_category"].tolist(), ["유통"])
        self.assertEqual(out.loc[0, "dongbaek_krw"], 2.0 * 100_000_000 * 3)  # 6~8월 = 3개월

    def test_period_total_not_multiplied(self) -> None:
        out = report_amounts(stats([("T", "P", "슈퍼", 2.0)]), MAPPING, basis(monthly=False))
        self.assertEqual(out.loc[0, "dongbaek_krw"], 2.0 * 100_000_000)

    def test_unmapped_category_raises(self) -> None:
        with self.assertRaises(ValueError):
            report_amounts(stats([("T", "P", "새업종", 1.0)]), MAPPING, basis(monthly=False))


class CardAmountsTest(unittest.TestCase):
    def test_missing_month_raises(self) -> None:
        d1 = pd.DataFrame({"ym": ["202506", "202507"], "category": ["유통", "유통"], "month_amount_krw": [1, 1]})
        with self.assertRaises(ValueError):
            card_amounts(d1, {"P": ["2025-06", "2025-08"]})


class ShareTableTest(unittest.TestCase):
    def test_share_and_combined_row(self) -> None:
        card = pd.DataFrame({"d1_category": ["유통", "음식료품"], "period": ["P", "P"], "card_krw": [300.0, 100.0]})
        dongbaek = pd.DataFrame({"d1_category": ["유통"], "period": ["P"], "dongbaek_krw": [3.0]})
        t = share_table(card, dongbaek, {"합": ["유통", "음식료품"]}).set_index("d1_category")
        self.assertEqual(t.loc["유통", "dongbaek_share_pct"], 1.0)
        self.assertTrue(pd.isna(t.loc["음식료품", "dongbaek_share_pct"]))  # 동백전 금액 없으면 비워 둠
        self.assertEqual(t.loc["합", "dongbaek_share_pct"], 0.75)
        self.assertEqual(t.loc["유통", "card_share_of_total_pct"], 75.0)


if __name__ == "__main__":
    unittest.main()
