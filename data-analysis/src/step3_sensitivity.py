"""STEP 3 1차 민감도 분석: 동백전 적립 추정액이 어느 입력값에 가장 크게 흔들리는지 본다.

연간 적립 총액(억 원) = 불가(A) 가게 연 카드 소비(조 원) × 카드 혜택률(%) × 이용자 비중(%)
1인당 연 적립(원)     = 불가(A) 가게 연 카드 소비 ÷ 부산 15세 이상 인구 × 카드 혜택률

- 입력값과 출처·이름표(사실/가정/추정/미확인): ref/step3_inputs.csv
- 결과는 공공데이터 기반 추정·시뮬레이션이다. 입력값 대부분이 "가정"이다.

사용법 (data-analysis 폴더에서): python src/step3_sensitivity.py
"""

from __future__ import annotations

import sys

import pandas as pd

from sensitivity import load_inputs, one_at_a_time, scenario_grid, values_at
from settings import load_config, resolve_path

JO, EOK = 1_000_000_000_000, 100_000_000  # 조 원, 억 원


def total_credit_eok(v: dict) -> float:
    """연간 적립 총액(억 원)."""
    return v["a_spend_total"] * JO * v["benefit_rate"] / 100 * v["user_share"] / 100 / EOK


def per_user_credit_won(v: dict) -> float:
    """1인당 연 적립(원). 동백 ONE 이용자의 A 소비가 15세 이상 평균과 같다고 가정한다."""
    return v["a_spend_total"] * JO / v["pop_15plus"] * v["benefit_rate"] / 100


def plot_tornado(tornado: pd.DataFrame, inputs: pd.DataFrame, path) -> None:
    """토네이도 그래프: 막대가 길수록 결과를 크게 흔드는 변수."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False
    data = tornado.iloc[::-1]
    base = data["result_mid"].iloc[0]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    for i, row in enumerate(data.itertuples()):
        lo, hi = sorted([row.result_low, row.result_high])
        ax.barh(i, hi - lo, left=lo, color="#c0392b" if i == len(data) - 1 else "#7f8c8d")
        ax.text(lo, i, f" {lo:,.0f}", va="center", ha="left", fontsize=9, color="white")
        ax.text(hi, i, f" {hi:,.0f}", va="center", ha="left", fontsize=9)
    labels = [f"{inputs.loc[k, 'name']}\n({inputs.loc[k, 'low']:g}~{inputs.loc[k, 'high']:g}{inputs.loc[k, 'unit']}, "
              f"{inputs.loc[k, 'status']})" for k in data["key"]]
    ax.set_yticks(range(len(data)), labels, fontsize=9)
    ax.axvline(base, color="black", linewidth=1)
    ax.set_xlabel("연간 동백전 적립 추정액 (억 원)")
    ax.set_title(f"민감도 분석 — 가운데 값 {base:,.0f}억 원 기준, 한 번에 하나만 바꿈 (추정·시뮬레이션)", fontsize=10)
    ax.set_xlim(0, data[["result_low", "result_high"]].max().max() * 1.12)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    spec = load_config()["step3"]
    inputs = load_inputs(resolve_path(spec["inputs_path"]))
    keys = spec["sensitivity_keys"]

    tornado = one_at_a_time(inputs, total_credit_eok, keys)
    tornado.insert(1, "name", tornado["key"].map(inputs["name"]))
    tornado["status"] = tornado["key"].map(inputs["status"])
    grid = scenario_grid(inputs, total_credit_eok, "benefit_rate", "user_share")
    per_user = scenario_grid(inputs, per_user_credit_won, "a_spend_total", "benefit_rate")
    budget = inputs.loc["cashback_budget_2024", "mid"]

    out = {k: resolve_path(p) for k, p in spec["outputs"].items()}
    for p in out.values():
        p.parent.mkdir(parents=True, exist_ok=True)
    tornado.round(2).to_csv(out["tornado"], index=False, encoding="utf-8-sig")
    long_grid = pd.concat([
        grid.stack().rename("value").reset_index().assign(metric="연간 적립 총액(억원)", fixed="a_spend_total=mid"),
        per_user.stack().rename("value").reset_index().assign(metric="1인당 연 적립(원)", fixed="user_share 무관"),
    ])
    long_grid["vs_cashback_budget_pct"] = (long_grid["value"] / budget * 100).where(
        long_grid["metric"].str.startswith("연간")).round(1)
    long_grid["note"] = "추정·시뮬레이션. 입력값 출처·이름표는 ref/step3_inputs.csv"
    long_grid.round(1).to_csv(out["grid"], index=False, encoding="utf-8-sig")
    plot_tornado(tornado, inputs, out["chart"])

    print("① 민감도 (연간 적립 총액, 억 원) — 막대가 긴 순서")
    print(tornado[["name", "result_low", "result_mid", "result_high", "ratio", "status"]].round(1).to_string(index=False))
    print(f"\n② 3×3 연간 적립 총액(억 원), A 소비 = {inputs.loc['a_spend_total', 'mid']:g}조 원")
    print(grid.round(1).to_string())
    print(f"   캐시백 예산 2024 대비(%):\n{(grid / budget * 100).round(1).to_string()}")
    print("\n③ 1인당 연 적립(원) — 이용자 비중과 무관")
    print(per_user.round(0).to_string())
    print(f"   1인당 월 적립(원):\n{(per_user / 12).round(0).to_string()}")
    print(f"\n저장: {', '.join(str(p.name) for p in out.values())}")


if __name__ == "__main__":
    main()
