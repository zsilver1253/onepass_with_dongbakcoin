"""STEP 1 업종 판정 근거표: D1 업종별 카드 소비 중 동백전 결제가 차지하는 비율(동백전 비중)을 계산한다.

동백전 비중 = 동백전 업종별 결제액(D10 보고서) ÷ 같은 업종·같은 기간 D1 카드 소비액
- 근거(basis)는 config.yaml 의 evidence.bases 에 보고서 표 단위로 적는다 (지금: 표3-6 연간, 표4-28 여름 3개월).
- 비중이 낮다고 "동백전 불가"라는 뜻은 아니다. 가게는 가능한데 사람들이 안 쓸 수도 있다.
- D1 의 범위(가게 위치 기준인지 사는 곳 기준인지, 동백전 포함 여부)가 확인되지 않아 "비중"으로만 해석한다 (확인 필요).

사용법 (data-analysis 폴더에서, clean_d1.py 실행 후): python src/industry_evidence.py
저장: config.yaml evidence.output_path (utf-8-sig)
"""

from __future__ import annotations

import sys

import pandas as pd

from data_go_kr import month_range
from settings import load_config, resolve_path

EOK = 100_000_000  # 1억 원
UNKNOWN = "불명"


def report_amounts(stats: pd.DataFrame, mapping: pd.DataFrame, basis: dict) -> pd.DataFrame:
    """보고서 업종별 결제액(억 원)을 D1 업종으로 묶어 기간별 합계(원)로 만든다.

    입력: 보고서 표(ref/dongbaek_stats_*.csv), 업종 연결표, basis 설정 하나
    출력: d1_category, period, dongbaek_krw (합계 행과 불명으로 연결된 업종은 제외)
    예외: 보고서 업종이 연결표에 없으면 ValueError (조용히 빠지지 않게)
    """
    rows = stats[(stats["table_id"] == basis["table_id"]) & stats["period"].isin(basis["periods"])
                 & (stats["category"] != basis.get("total_category"))]
    missing = set(rows["category"]) - set(mapping["report_category"])
    if missing:
        raise ValueError(f"{basis['table_id']} 업종이 연결표에 없습니다: {sorted(missing)}")
    rows = rows.merge(mapping, left_on="category", right_on="report_category")
    rows = rows[rows["d1_category"] != UNKNOWN]
    out = rows.groupby(["d1_category", "period"])["value"].sum().reset_index()
    out["dongbaek_krw"] = out["value"] * EOK
    if basis["value_is_monthly_avg"]:
        months = {p: len(month_range(*rng)) for p, rng in basis["periods"].items()}
        out["dongbaek_krw"] *= out["period"].map(months)
    return out.drop(columns="value")


def card_amounts(d1: pd.DataFrame, periods: dict) -> pd.DataFrame:
    """정제된 D1 에서 업종 × 기간 별 카드 소비 합계(원)를 구한다.

    예외: 기간 안의 달이 D1 에 하나라도 없으면 ValueError (일부 달만 더해지는 것 방지)
    """
    parts = []
    for period, rng in periods.items():
        months = month_range(*rng)
        lacking = sorted(set(months) - set(d1["ym"]))
        if lacking:
            raise ValueError(f"D1 에 {period} 기간의 달이 없습니다: {lacking}")
        part = d1[d1["ym"].isin(months)].groupby("category")["month_amount_krw"].sum().reset_index()
        part["period"] = period
        parts.append(part)
    out = pd.concat(parts, ignore_index=True)
    return out.rename(columns={"category": "d1_category", "month_amount_krw": "card_krw"})


def share_table(card: pd.DataFrame, dongbaek: pd.DataFrame, combine: dict) -> pd.DataFrame:
    """업종 × 기간 별 동백전 비중(%)과 카드 소비 중 업종 비중(%)을 계산한다.

    combine: {"새 행 이름": [합칠 D1 업종, ...]} 카드·동백전 금액을 각각 더한 행을 추가한다.
    동백전 금액이 없는 업종은 비중을 비워 둔다 (0 으로 채우지 않음).
    """
    table = card.merge(dongbaek, on=["d1_category", "period"], how="left")
    total = card.groupby("period")["card_krw"].sum()
    table["card_share_of_total_pct"] = (table["card_krw"] / table["period"].map(total) * 100).round(1)
    for name, members in combine.items():
        part = table[table["d1_category"].isin(members)]
        summed = part.groupby("period").agg(card_krw=("card_krw", "sum"),
                                            dongbaek_krw=("dongbaek_krw", lambda s: s.sum(min_count=1)),
                                            card_share_of_total_pct=("card_share_of_total_pct", "sum"))
        table = pd.concat([table, summed.reset_index().assign(d1_category=name)], ignore_index=True)
    table["dongbaek_share_pct"] = (table["dongbaek_krw"] / table["card_krw"] * 100).round(2)
    return table


def build_evidence(config: dict, d1: pd.DataFrame) -> pd.DataFrame:
    """config 의 모든 basis 에 대해 비중 표를 만들어 한 표로 합친다."""
    stats = pd.read_csv(resolve_path(config["ref"]["report_stats"]), encoding="utf-8-sig", dtype={"period": str})
    tables = []
    for basis in config["evidence"]["bases"]:
        mapping = pd.read_csv(resolve_path(config["ref"][basis["category_map"]]), encoding="utf-8-sig")
        table = share_table(card_amounts(d1, basis["periods"]), report_amounts(stats, mapping, basis),
                            basis["combine"])
        table.insert(0, "basis", basis["name"])
        table["source"] = (f"카드: D1(공공데이터포털 15164434) / 동백전: D10 부산연구원(2026.3) {basis['table_id']}"
                           f", 업종 연결은 {config['ref'][basis['category_map']]}(가정)")
        tables.append(table)
    out = pd.concat(tables, ignore_index=True)
    cols = ["basis", "period", "d1_category", "card_krw", "dongbaek_krw", "dongbaek_share_pct",
            "card_share_of_total_pct", "source"]
    return out[cols].sort_values(["basis", "period", "dongbaek_share_pct"], ascending=[True, True, False])


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    config = load_config()
    d1 = pd.read_parquet(resolve_path(config["datasets"]["d1"]["processed_path"]))
    table = build_evidence(config, d1)
    out = resolve_path(config["evidence"]["output_path"])
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False, encoding="utf-8-sig")
    show = table.pivot_table(index="d1_category", columns=["basis", "period"], values="dongbaek_share_pct")
    print("동백전 비중(%) — 업종별 카드 소비 중 동백전 결제 비율")
    print(show.round(2).to_string())


if __name__ == "__main__":
    main()
