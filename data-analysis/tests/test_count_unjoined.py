"""count_unjoined 테스트. 실행: data-analysis 폴더에서 python -m unittest discover tests"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from count_unjoined import (  # noqa: E402
    STATUS_EXACT, STATUS_FUZZY, STATUS_NONE, classify_stores, match, normalize_name,
    parse_gu, parse_road_address, prepare_merchants, store_keys, strip_branch, summarize,
)

LEGAL = ["주식회사", "㈜", "(주)"]
ALIASES = {"씨유": "cu"}
GU = ["연제구", "사하구", "부산진구"]


class NormalizeTest(unittest.TestCase):
    def test_name_rules(self) -> None:
        self.assertEqual(normalize_name("(주)씨유 연산점", LEGAL, ALIASES), "cu연산점")
        self.assertEqual(normalize_name("메이브(Maeve)", LEGAL, ALIASES), "메이브")
        self.assertEqual(normalize_name("Fit Gym 주식회사", LEGAL, ALIASES), "fitgym")
        self.assertEqual(normalize_name(None, LEGAL, ALIASES), "")

    def test_strip_branch(self) -> None:
        self.assertEqual(strip_branch("온도카페 연산점"), "온도카페")
        self.assertEqual(strip_branch("온도카페"), "온도카페")
        self.assertEqual(strip_branch("빵집 본관"), "빵집 본관")

    def test_road_address(self) -> None:
        self.assertEqual(parse_road_address("부산 연제구 중앙대로 1120번길 15, 2층"),
                         ("연제구", "중앙대로1120번길", "15", "0"))
        self.assertEqual(parse_road_address("부산 동래구 반송로 241-1 ,1층 (안락동)"), ("동래구", "반송로", "241", "1"))
        self.assertEqual(parse_road_address("부산 부산진구 중앙대로 지하 717"), ("부산진구", "중앙대로", "717", "0"))
        self.assertIsNone(parse_road_address("부산 사상구 모라3동"))
        self.assertIsNone(parse_road_address(None))

    def test_parse_gu(self) -> None:
        self.assertEqual(parse_gu("부산 사하 감천로 1", GU), "사하구")
        self.assertIsNone(parse_gu("부산", GU))


class ClassifyTest(unittest.TestCase):
    EMAP = pd.DataFrame({
        "상권업종소분류코드": ["S1", "S2", "S3"],
        "판정": ["애매", "사용 가능", "사용 불가"],
        "애매유형": ["상호판단", None, None],
        "키워드_미해당시": ["사용 가능", None, None],
    })
    RULES = pd.DataFrame({"규칙ID": ["R01", "R04"], "적용소분류": ["*", "*"],
                          "상호패턴": ["이마트(?!24)", "아울렛"], "판정": ["사용 불가", "애매"]})

    def run_classify(self, names: list[str], codes: list[str]) -> pd.DataFrame:
        stores = pd.DataFrame({"상호명": names, "상권업종소분류코드": codes, "상권업종소분류명": ["x"] * len(names)})
        return classify_stores(stores, self.EMAP, self.RULES, LEGAL, ALIASES)

    def test_keyword_overrides_and_default(self) -> None:
        out = self.run_classify(["이마트 해운대", "이마트24 연산", "동네슈퍼", "옷 아울렛", "주점"],
                                ["S1", "S1", "S1", "S2", "S3"])
        self.assertEqual(out["최종판정"].tolist(), ["사용 불가", "사용 가능", "사용 가능", "애매", "사용 불가"])
        self.assertEqual(out.loc[3, "최종애매유형"], "상호판단")

    def test_unknown_category_raises(self) -> None:
        with self.assertRaises(ValueError):
            self.run_classify(["가게"], ["S9"])


class MatchTest(unittest.TestCase):
    def setUp(self) -> None:
        stores = pd.DataFrame({
            "상가업소번호": ["a", "b", "c", "d"],
            "상호명": ["온도카페", "행복미용실", "효원떡방", "새가게"],
            "지점명": ["연산점", None, None, None],
            "시군구명": ["연제구", "연제구", "사하구", "사하구"],
            "도로명": ["부산광역시 연제구 중앙대로"] * 2 + ["부산광역시 사하구 감천로"] * 2,
            "건물본번지": ["10", "10", "5", "7"], "건물부번지": [None] * 4,
        })
        merchants = pd.DataFrame({"가맹점 명": ["온도카페 연산점", "행복미용실헤어", "효원떡방", "효원떡방"],
                                  "주소": ["부산 연제구 중앙대로 10", "부산 연제구 중앙대로 10",
                                         "부산 사하구", "부산광역시 사하구"]})
        self.s = store_keys(stores, LEGAL, ALIASES)
        self.m = prepare_merchants(merchants, LEGAL, ALIASES, GU)

    def test_dedup_merges_busan_prefix(self) -> None:
        self.assertEqual(len(self.m), 3)

    def test_three_rules(self) -> None:
        result, matched = match(self.s, self.m, 0.8, 3)
        status = dict(zip(result["상가업소번호"], result["매칭상태"]))
        self.assertEqual(status, {"a": STATUS_EXACT, "b": STATUS_FUZZY, "c": STATUS_FUZZY, "d": STATUS_NONE})
        self.assertEqual(len(matched), 3)

    def test_summarize_ranges(self) -> None:
        result, _ = match(self.s, self.m, 0.8, 3)
        df = result.assign(최종판정="사용 가능", 최종애매유형="")
        low = summarize(df, {"name": "하한", "include_ambiguous": [], "fuzzy_as_joined": True})
        high = summarize(df, {"name": "상한", "include_ambiguous": [], "fuzzy_as_joined": False})
        self.assertEqual((low.loc[0, "미가입"], high.loc[0, "미가입"]), (1, 3))


if __name__ == "__main__":
    unittest.main()
