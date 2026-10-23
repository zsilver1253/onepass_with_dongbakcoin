"""공통 민감도 분석 도구: 입력값 표(low/mid/high)와 계산 함수만 주면 어느 모델에도 쓸 수 있다.

- one_at_a_time: 한 번에 변수 하나만 low/high 로 바꾸고 나머지는 mid 로 둔다 (토네이도 그래프용)
- scenario_grid: 변수 두 개의 low/mid/high 를 모두 조합한다 (3×3 표)
"""

from __future__ import annotations

from typing import Callable

import pandas as pd

LEVELS = ("low", "mid", "high")
Model = Callable[[dict], float]


def load_inputs(path) -> pd.DataFrame:
    """입력값 표를 읽는다. 필수 열: key, role, low, mid, high, unit, status

    예외: low ≤ mid ≤ high 가 아닌 행이 있으면 ValueError (범위를 거꾸로 적는 실수 방지)
    """
    table = pd.read_csv(path, encoding="utf-8-sig")
    bad = table[~((table["low"] <= table["mid"]) & (table["mid"] <= table["high"]))]
    if not bad.empty:
        raise ValueError(f"low ≤ mid ≤ high 가 아닌 입력값: {bad['key'].tolist()}")
    return table.set_index("key")


def values_at(inputs: pd.DataFrame, level: str = "mid", **overrides: float) -> dict:
    """모든 변수를 level 값으로 두고, overrides 로 일부만 바꾼 값 묶음을 만든다."""
    values = inputs[level].to_dict()
    values.update(overrides)
    return values


def one_at_a_time(inputs: pd.DataFrame, model: Model, keys: list[str]) -> pd.DataFrame:
    """keys 의 변수를 하나씩 low/high 로 바꿔 결과 변화를 잰다.

    출력: key, low_value, high_value, result_low, result_mid, result_high, swing(= |high - low| 결과 차이), ratio(= 큰 결과 ÷ 작은 결과)
    swing 이 큰 순서로 정렬 = 결과를 가장 크게 흔드는 변수가 위
    """
    base = model(values_at(inputs))
    rows = []
    for key in keys:
        lo = model(values_at(inputs, **{key: inputs.loc[key, "low"]}))
        hi = model(values_at(inputs, **{key: inputs.loc[key, "high"]}))
        small, big = sorted([lo, hi])
        rows.append({"key": key, "low_value": inputs.loc[key, "low"], "high_value": inputs.loc[key, "high"],
                     "result_low": lo, "result_mid": base, "result_high": hi, "swing": big - small,
                     "ratio": big / small if small else float("inf")})
    return pd.DataFrame(rows).sort_values("swing", ascending=False).reset_index(drop=True)


def scenario_grid(inputs: pd.DataFrame, model: Model, row_key: str, col_key: str) -> pd.DataFrame:
    """두 변수의 low/mid/high 를 모두 조합한 표를 만든다. 나머지 변수는 mid."""
    grid = {}
    for col_level in LEVELS:
        col_value = inputs.loc[col_key, col_level]
        grid[col_value] = {
            inputs.loc[row_key, row_level]: model(values_at(inputs, **{row_key: inputs.loc[row_key, row_level],
                                                                       col_key: col_value}))
            for row_level in LEVELS
        }
    table = pd.DataFrame(grid)
    table.index.name, table.columns.name = row_key, col_key
    return table
