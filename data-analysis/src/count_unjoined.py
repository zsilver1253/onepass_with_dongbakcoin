"""동백전 "가입 가능한데 미가입" 가게 수 추정.

상가(상권)정보 가게 목록과 동백전 가맹점 명단을 상호·주소로 맞춰 보고,
사용 불가 업종을 뺀 가게 중 가맹점 명단에 없는 가게 수를 범위(하한~상한)로 낸다.
결과는 공공데이터 기반 **추정**이다.

실행: data-analysis 폴더에서  python src/count_unjoined.py
설정: config.yaml 의 unjoined 항목, 분류표: ref/exclusion_map.csv, ref/exclusion_keywords.csv

정규화 규칙
- 상호: 영문 소문자 → 괄호와 괄호 안 내용 삭제 → 법인 표기(주식회사, ㈜ 등) 삭제
        → 한글·영문·숫자 외 문자(공백 포함) 삭제 → 브랜드 별칭 통일(씨유→cu 등)
- 지점명: 상가정보는 '상호명'과 '상호명+지점명' 두 가지, 가맹점은 '전체'와
        '마지막 띄어쓰기 뒤가 ~점이면 그 앞까지' 두 가지로 비교한다.
- 주소: '부산'/'부산광역시' 표기 통일 → 구·군, 도로명(…로/…길, '번길' 포함), 건물 본번-부번만 뽑는다.
        층·호·괄호(동 이름)는 쓰지 않는다. 도로명이 없는 주소(지번 주소)는 매칭하지 않는다.
"""

from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd

from settings import load_config, resolve_path

STATUS_EXACT, STATUS_FUZZY, STATUS_NONE = "확실 일치", "애매 일치", "불일치"
VERDICT_OK, VERDICT_BAN, VERDICT_AMBIG = "사용 가능", "사용 불가", "애매"
VERDICT_RANK = {VERDICT_OK: 0, VERDICT_AMBIG: 1, VERDICT_BAN: 2}  # 규칙이 겹치면 더 엄격한 쪽

ROAD_PATTERN = re.compile(r"(\S+(?:로|길))\s*(\d+)(?:-(\d+))?")
ROAD_GAP = re.compile(r"(로|길)\s+(\d+번?[가-힣]?길)")  # '중앙대로 1120번길' → '중앙대로1120번길'
UNDERGROUND = re.compile(r"(로|길)\s+지하\s*(?=\d)")       # '중앙대로 지하 717' → '중앙대로 717'
BRACKETS = re.compile(r"\([^)]*\)|\[[^\]]*\]|\{[^}]*\}|（[^）]*）")
NON_WORD = re.compile(r"[^0-9a-z가-힣]")


# ---------- 정규화 ----------

def normalize_name(name: str, legal_terms: list[str], aliases: dict[str, str]) -> str:
    """상호를 비교용 문자열로 바꾼다. 예: '(주)씨유 연산점' → 'cu연산점'"""
    if not isinstance(name, str):
        return ""
    s = name.lower()
    for term in legal_terms:  # '(주)'처럼 괄호가 붙은 표기는 괄호 삭제 전에 지운다
        s = s.replace(term.lower(), "")
    s = NON_WORD.sub("", BRACKETS.sub("", s))
    for src, dst in aliases.items():
        s = s.replace(src, dst)
    return s


def strip_branch(name: str) -> str:
    """띄어쓰기로 붙은 지점명을 뗀다. 예: '온도카페 연산점' → '온도카페'. 해당 없으면 원래 값."""
    if not isinstance(name, str):
        return ""
    parts = name.strip().rsplit(" ", 1)
    return parts[0] if len(parts) == 2 and parts[1].endswith("점") else name


def parse_road_address(addr: str) -> tuple[str, str, str, str] | None:
    """도로명주소에서 (구·군, 도로명, 본번, 부번)을 뽑는다. 부번이 없으면 '0'.

    예: '부산 연제구 중앙대로 1120번길 15, 2층' → ('연제구', '중앙대로1120번길', '15', '0')
    도로명을 못 찾으면 None.
    """
    if not isinstance(addr, str):
        return None
    tokens = addr.replace(",", " ").split()
    if len(tokens) < 3 or not tokens[0].startswith("부산"):
        return None
    gu = tokens[1]
    rest = ROAD_GAP.sub(r"\1\2", UNDERGROUND.sub(r"\1 ", " ".join(tokens[2:])))
    m = ROAD_PATTERN.search(rest)
    return (gu, m.group(1), m.group(2), m.group(3) or "0") if m else None


def parse_gu(addr: str, gu_names: list[str]) -> str | None:
    """주소의 두 번째 단어에서 구·군 이름을 뽑는다. '사하'처럼 '구'가 빠져도 맞춘다. 못 찾으면 None."""
    tokens = addr.split() if isinstance(addr, str) else []
    if len(tokens) < 2:
        return None
    token = tokens[1]
    return next((g for g in (token, token + "구", token + "군") if g in gu_names), None)


def similarity(a: str, b: str, contain_min_len: int) -> float:
    """두 상호의 유사도(0~1). 짧은 쪽이 긴 쪽에 통째로 들어 있으면 1."""
    if not a or not b:
        return 0.0
    short, long_ = sorted((a, b), key=len)
    if len(short) >= contain_min_len and short in long_:
        return 1.0
    return SequenceMatcher(None, a, b).ratio()


# ---------- 입력 ----------

def find_file(path: Path) -> Path:
    """한글 파일명이 macOS 방식(NFD)으로 저장돼 있어도 찾는다. 없으면 FileNotFoundError."""
    if path.exists():
        return path
    target = unicodedata.normalize("NFC", path.name)
    for p in path.parent.iterdir():
        if unicodedata.normalize("NFC", p.name) == target:
            return p
    raise FileNotFoundError(path)


def read_csv(path: Path) -> pd.DataFrame:
    """CSV 를 문자열로 읽는다. UTF-8(BOM 포함)이 안 되면 cp949 로 읽는다. 원본은 고치지 않는다."""
    path = find_file(path)
    try:
        return pd.read_csv(path, encoding="utf-8-sig", dtype=str)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="cp949", dtype=str)


def inventory(df: pd.DataFrame, label: str, reference: str) -> pd.DataFrame:
    """파일 기본 점검표: 컬럼별 결측 수와 고유값 수."""
    return pd.DataFrame({
        "파일": label, "기준시점": reference, "행수": len(df), "컬럼": df.columns,
        "결측수": df.isna().sum().values, "고유값수": df.nunique().values,
    })


# ---------- 업종 분류 ----------

def classify_stores(stores: pd.DataFrame, emap: pd.DataFrame, rules: pd.DataFrame,
                    legal_terms: list[str], aliases: dict[str, str]) -> pd.DataFrame:
    """가게마다 최종 판정(사용 가능/불가/애매)과 애매 유형, 적용된 규칙을 붙인다.

    1) 소분류 판정을 붙인다. 2) 상호 키워드 규칙이 맞으면 더 엄격한 판정으로 바꾼다.
    3) '상호판단' 소분류에서 키워드가 하나도 안 맞으면 '키워드_미해당시' 값을 쓴다.
    """
    cols = ["상권업종소분류코드", "판정", "애매유형", "키워드_미해당시"]
    out = stores.merge(emap[cols], on="상권업종소분류코드", how="left", validate="m:1")
    if out["판정"].isna().any():
        missing = out.loc[out["판정"].isna(), "상권업종소분류명"].unique()
        raise ValueError(f"exclusion_map 에 없는 소분류: {list(missing)}")
    out["애매유형"] = out["애매유형"].fillna("")
    name_norm = out["상호명"].map(lambda s: normalize_name(s, legal_terms, aliases))
    out["최종판정"], out["최종애매유형"], out["적용규칙"] = out["판정"], out["애매유형"], ""

    for rule in rules.itertuples(index=False):
        scope = (pd.Series(True, index=out.index) if rule.적용소분류 == "*"
                 else out["상권업종소분류명"].isin(rule.적용소분류.split("|")))
        hit = scope & name_norm.str.contains(rule.상호패턴, regex=True)
        stricter = hit & (out["최종판정"].map(VERDICT_RANK) < VERDICT_RANK[rule.판정])
        out.loc[stricter, "최종판정"] = rule.판정
        out.loc[stricter, "최종애매유형"] = "상호판단" if rule.판정 == VERDICT_AMBIG else ""
        out.loc[hit, "적용규칙"] = (out.loc[hit, "적용규칙"] + " " + rule.규칙ID).str.strip()

    no_rule = (out["애매유형"] == "상호판단") & (out["적용규칙"] == "")
    out.loc[no_rule, "최종판정"] = out.loc[no_rule, "키워드_미해당시"]
    out.loc[no_rule & (out["최종판정"] != VERDICT_AMBIG), "최종애매유형"] = ""
    return out.drop(columns=["판정", "애매유형", "키워드_미해당시"])


# ---------- 가맹점 정리 ----------

def prepare_merchants(merchants: pd.DataFrame, legal_terms: list[str], aliases: dict[str, str],
                      gu_names: list[str]) -> pd.DataFrame:
    """가맹점 명단을 정규화하고 '상호+주소' 중복을 뺀다.

    주소는 '부산광역시'→'부산', 연속 공백 정리만 한 뒤 비교한다.
    구·군은 건물번호가 없는 주소에서도 뽑는다(3순위 매칭용).
    """
    m = merchants.rename(columns={"가맹점 명": "name", "주소": "addr"}).copy()
    m["addr_clean"] = m["addr"].str.replace("부산광역시", "부산", regex=False).str.split().str.join(" ")
    m["name_clean"] = m["name"].str.split().str.join(" ")
    m["prefix_full"] = m["addr"].str.startswith("부산광역시")
    m = m.drop_duplicates(["name_clean", "addr_clean"]).reset_index(drop=True)
    m["merchant_id"] = m.index
    keys = m["addr_clean"].map(parse_road_address)
    m["addr_key"] = keys.map(lambda k: "|".join(k) if k else None)
    m["gu"] = m["addr_clean"].map(lambda a: parse_gu(a, gu_names))
    m["name_full"] = m["name_clean"].map(lambda s: normalize_name(s, legal_terms, aliases))
    m["name_base"] = m["name_clean"].map(lambda s: normalize_name(strip_branch(s), legal_terms, aliases))
    return m


def store_keys(stores: pd.DataFrame, legal_terms: list[str], aliases: dict[str, str]) -> pd.DataFrame:
    """상가정보에 비교용 주소 키와 상호 두 가지(상호명, 상호명+지점명)를 붙인다."""
    s = stores.copy()
    road = s["도로명"].str.split().str[-1]
    sub = s["건물부번지"].fillna("0")
    s["addr_key"] = s["시군구명"] + "|" + road + "|" + s["건물본번지"] + "|" + sub
    s["gu"] = s["시군구명"]
    s["name_base"] = s["상호명"].map(lambda x: normalize_name(x, legal_terms, aliases))
    s["name_full"] = (s["상호명"].fillna("") + s["지점명"].fillna("")).map(
        lambda x: normalize_name(x, legal_terms, aliases))
    return s


# ---------- 매칭 ----------

def name_variants(df: pd.DataFrame, id_col: str, key: str = "addr_key") -> pd.DataFrame:
    """(id, key, name) 긴 표. 상호 두 가지를 모두 넣고 빈 이름·빈 키는 뺀다."""
    long = pd.concat([df[[id_col, key, c]].rename(columns={c: "name"}) for c in ("name_base", "name_full")])
    return long[(long["name"] != "") & long[key].notna()].drop_duplicates()


def unique_name_pairs(stores: pd.DataFrame, merchants: pd.DataFrame, min_len: int) -> pd.DataFrame:
    """같은 구 안에서 정규화 상호가 양쪽 모두 한 곳뿐인 (가게, 가맹점) 쌍. 주소가 달라도 같은 가게로 본다.

    '비누'처럼 짧은 이름은 우연히 겹칠 수 있어 min_len 글자 미만은 쓰지 않는다.
    """
    sv, mv = name_variants(stores, "상가업소번호", "gu"), name_variants(merchants, "merchant_id", "gu")
    sv, mv = sv[sv["name"].str.len() >= min_len], mv[mv["name"].str.len() >= min_len]
    sv = sv[sv.groupby(["gu", "name"])["상가업소번호"].transform("nunique") == 1]
    mv = mv[mv.groupby(["gu", "name"])["merchant_id"].transform("nunique") == 1]
    return sv.merge(mv, on=["gu", "name"]).drop_duplicates("상가업소번호")


def match(stores: pd.DataFrame, merchants: pd.DataFrame, sim_min: float, contain_min: int) -> tuple[pd.DataFrame, set]:
    """가게별 매칭 결과와, 어느 가게와든 맞은 가맹점 id 집합을 돌려준다.

    1순위(확실 일치): 같은 주소 키 + 정규화 상호가 같음
    2순위(애매 일치): 같은 주소 키(같은 건물) + 상호 유사도 ≥ sim_min
    3순위(애매 일치): 같은 구 + 정규화 상호가 같고 그 이름이 구 안에서 양쪽 모두 한 곳뿐
                     (가맹점 주소에 건물번호가 없거나 옛 주소로 적힌 경우를 잡기 위함)
    한 가게에 여러 가맹점이 맞아도 '가입'으로 하나만 센다(일대일 배정 안 함).
    """
    sv, mv = name_variants(stores, "상가업소번호"), name_variants(merchants, "merchant_id")
    exact = sv.merge(mv, on=["addr_key", "name"])
    exact_best = exact.drop_duplicates("상가업소번호").set_index("상가업소번호")["merchant_id"]

    rest = stores.loc[~stores["상가업소번호"].isin(exact_best.index) & stores["addr_key"].notna(),
                      ["상가업소번호", "addr_key", "name_base", "name_full"]]
    pairs = rest.merge(merchants[["merchant_id", "addr_key", "name_base", "name_full"]],
                       on="addr_key", suffixes=("_s", "_m"))
    pairs["score"] = [
        max(similarity(a, b, contain_min) for a in (sb, sf) for b in (mb, mf))
        for sb, sf, mb, mf in zip(pairs["name_base_s"], pairs["name_full_s"], pairs["name_base_m"], pairs["name_full_m"])
    ]
    fuzzy = pairs[pairs["score"] >= sim_min]
    fuzzy_best = fuzzy.sort_values("score", ascending=False).drop_duplicates("상가업소번호").set_index("상가업소번호")

    done = set(exact_best.index) | set(fuzzy_best.index)
    by_gu = unique_name_pairs(stores[~stores["상가업소번호"].isin(done)], merchants, contain_min).set_index("상가업소번호")

    result = pd.DataFrame({"상가업소번호": stores["상가업소번호"]})
    result["매칭상태"], result["매칭규칙"] = STATUS_NONE, ""
    for rule, status, ids in (("3 같은 구+유일 상호", STATUS_FUZZY, by_gu.index),
                              ("2 같은 건물+유사 상호", STATUS_FUZZY, fuzzy_best.index),
                              ("1 같은 주소+같은 상호", STATUS_EXACT, exact_best.index)):
        hit = result["상가업소번호"].isin(ids)
        result.loc[hit, "매칭상태"], result.loc[hit, "매칭규칙"] = status, rule
    sid = result["상가업소번호"]
    mid = sid.map(exact_best).fillna(sid.map(fuzzy_best["merchant_id"])).fillna(sid.map(by_gu["merchant_id"]))
    result["가맹점id"] = mid.astype("Int64")
    result["상호유사도"] = sid.map(fuzzy_best["score"]).round(3)
    result.loc[result["매칭규칙"].str.startswith(("1", "3")), "상호유사도"] = 1.0
    matched_merchants = set(exact["merchant_id"]) | set(fuzzy["merchant_id"]) | set(by_gu["merchant_id"])
    return result, matched_merchants


# ---------- 집계 ----------

def scenario_mask(df: pd.DataFrame, include_ambiguous: list[str]) -> pd.Series:
    """시나리오의 '가입 가능 가게' 범위."""
    return (df["최종판정"] == VERDICT_OK) | ((df["최종판정"] == VERDICT_AMBIG) & df["최종애매유형"].isin(include_ambiguous))


def summarize(df: pd.DataFrame, scenario: dict, by: str | None = None) -> pd.DataFrame:
    """가입 가능 가게 수, 가맹점 수, 미가입 수, 미가입 비율 (by 가 있으면 그룹별)."""
    target = df[scenario_mask(df, scenario["include_ambiguous"])]
    joined_status = [STATUS_EXACT, STATUS_FUZZY] if scenario["fuzzy_as_joined"] else [STATUS_EXACT]
    joined = target["매칭상태"].isin(joined_status)
    grp = target.assign(joined=joined).groupby(by) if by else target.assign(joined=joined).groupby(lambda _: "전체")
    out = grp["joined"].agg(가입가능=("size"), 가맹점=("sum")).reset_index().rename(columns={"index": by or "구분"})
    out["미가입"] = out["가입가능"] - out["가맹점"]
    out["미가입비율"] = (out["미가입"] / out["가입가능"]).round(3)
    out.insert(0, "시나리오", scenario["name"])
    return out


def sample_check(df: pd.DataFrame, status: str, n: int, seed: int) -> pd.DataFrame:
    """가입 가능(애매 포함) 가게 중 해당 매칭상태에서 n건을 뽑는다."""
    pool = df[(df["최종판정"] != VERDICT_BAN) & (df["매칭상태"] == status)]
    return pool.sample(min(n, len(pool)), random_state=seed)


# ---------- 보고서 ----------

def md_table(df: pd.DataFrame) -> str:
    """DataFrame → 마크다운 표 (tabulate 없이)."""
    head = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "---|" * len(df.columns)
    rows = ["| " + " | ".join(f"{v:,}" if isinstance(v, int) else str(v) for v in r) + " |"
            for r in df.astype(object).itertuples(index=False)]
    return "\n".join([head, sep, *rows])


def write_report(path: Path, ctx: dict) -> None:
    """report.md 를 쓴다. 숫자는 모두 ctx 에서 온다."""
    c = ctx
    lines = [
        "# 동백전 '가입 가능한데 미가입' 가게 수 추정",
        "",
        f"- 작성: `src/count_unjoined.py` 자동 생성. 모든 숫자는 공공데이터 기반 **추정**이에요(실측 아님).",
        f"- 가게 목록: 소상공인시장진흥공단 상가(상권)정보 부산 ({c['store_ref']})",
        f"- 가맹점 명단: 부산광역시 동백전 가맹점 현황 ({c['merchant_ref']}, 공공데이터포털 15088786)",
        f"- 업종 분류표 `ref/exclusion_map.csv`, 상호 키워드 `ref/exclusion_keywords.csv`는 **사람 검토 전 초안**이에요.",
        "",
        "## 1. 결과 (범위)",
        "",
        md_table(c["summary"]),
        "",
        "- 하한: 애매 업종 제외 + 애매 일치를 가입으로 계산. 상한: 애매 업종 포함 + 애매 일치를 미가입으로 계산.",
        "- B2B 포함/제외: 기업 상대 업종(광고·설계·법무·세무·인력·청소 등)을 '가입 가능'에 넣었는지 여부.",
        "",
        "## 2. 기본 점검",
        "",
        md_table(c["files"]),
        "",
        f"- 가맹점 명단 상호+주소 중복 제거: {c['merchant_rows']:,}행 → **{c['merchant_dedup']:,}곳** "
        f"(팀 V1.3 집계 {c['merchant_prev']:,}곳, 차이 {c['merchant_dedup'] - c['merchant_prev']:+,}).",
        f"  주소의 '부산광역시'/'부산' 표기를 하나로 맞춘 뒤 중복을 뺐기 때문에 차이가 날 수 있어요.",
        f"- 가맹점 주소 중 도로명·건물번호를 못 읽은 곳: {c['merchant_unparsed']:,}곳 (매칭 대상에서 빠짐).",
        "",
        "## 3. 업종 분류 (가게 수)",
        "",
        md_table(c["verdicts"]),
        "",
        "## 4. 매칭 결과 (가게 기준, 전체 상가정보)",
        "",
        md_table(c["match_counts"]),
        "",
        md_table(c["rule_counts"]),
        "",
        "## 5. 구별 (하한·상한 양 끝)",
        "",
        md_table(c["by_gu"]),
        "",
        "## 6. 업종 대분류별 (하한·상한 양 끝)",
        "",
        md_table(c["by_cat"]),
        "",
        "## 7. 반대 방향 확인",
        "",
        f"- 가맹점 명단 {c['merchant_dedup']:,}곳 중 상가정보의 어떤 가게와도 맞지 않은 곳: "
        f"**{c['merchant_unmatched']:,}곳** ({c['merchant_unmatched'] / c['merchant_dedup']:.1%}).",
        "- 상가정보에 없는 업종(개인택시, 무점포, 노점 등)과 주소·상호 표기 차이가 함께 섞여 있어요.",
        "",
        "## 8. 정규화·매칭 규칙",
        "",
        "- 상호: 소문자 → 괄호 삭제 → 법인 표기(주식회사, ㈜ 등) 삭제 → 공백·특수문자 삭제 → 브랜드 별칭 통일(씨유→cu 등).",
        "- 지점명: 상가정보는 '상호명', '상호명+지점명' 두 가지 / 가맹점은 '전체', '띄어쓰기 뒤 ~점을 뗀 이름' 두 가지로 비교.",
        "- 주소: 구·군 + 도로명(번길 포함) + 건물 본번-부번. 층·호·동 이름은 쓰지 않음.",
        f"- 확실 일치: 같은 주소 + 같은 상호. 애매 일치: 같은 주소(건물) + 상호 유사도 ≥ {c['sim_min']} "
        f"(또는 {c['contain_min']}글자 이상 상호가 다른 쪽에 통째로 포함).",
        f"- 3순위(애매 일치): 같은 구 안에서 정규화 상호({c['contain_min']}글자 이상)가 양쪽 모두 한 곳뿐이면 주소가 달라도 같은 가게로 봐요. "
        "가맹점 주소 중 건물번호 없이 '부산 사상구'처럼 구·동까지만 적힌 곳이 많아서 넣은 보완 규칙이에요.",
        "- 가맹점 명단에 좌표가 없어서 '반경 30m' 매칭은 하지 못했어요. 같은 건물(도로명+건물번호)까지만 봐요.",
        "",
        "## 9. 한계",
        "",
        f"- **시점 차이:** 가맹점 명단은 {c['merchant_ref']}, 상가정보는 {c['store_ref']}이에요. "
        "그 사이 새로 가입하거나 문을 닫은 가게는 맞지 않아요.",
        "- **상가정보는 전체 사업체가 아니에요.** 소상공인 전체(약 50만 개, 부산연구원 표 3-4 역산)보다 훨씬 적어서, "
        "팀의 '가입 비중 31.9%'(분모 소상공인)와 이 결과의 비율은 분모가 달라 직접 비교할 수 없어요.",
        "- 편의점·주유소의 직영 여부, SSM·대규모점포 입점 여부는 상호 키워드로만 판단했어요.",
        "- 상호가 많이 다르게 적힌 가게(상가정보는 법인명, 가맹점은 간판명 등)는 '불일치'로 남아 미가입이 부풀 수 있어요.",
        "- 이 결과는 가게 수 추정일 뿐이고, 동백전 결제나 지역 매출 변화에 대한 해석이 아니에요.",
        "",
        "## 10. 이전 팀 결과와 비교",
        "",
        "- 이전 매칭 일치율 31.4%는 매칭 규칙·분모를 문서에서 찾지 못해 직접 비교할 수 없어요. "
        "이번 결과는 분모가 '상가정보 중 가입 가능 업종'이고, 상호+도로명 건물번호 기준이에요.",
        "",
        "## 11. 손으로 확인할 것",
        "",
        f"- `{c['out_dir']}/manual_check.csv`: 불일치·애매 일치 각 {c['n']}건 (seed {c['seed']}). 동백전 앱에서 가맹 여부 확인.",
        "- `ref/exclusion_map.csv`, `ref/exclusion_keywords.csv` 판정 검토 후 다시 실행.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------- 실행 ----------

def main() -> None:
    config = load_config()
    cfg = config["unjoined"]
    legal, aliases = cfg["legal_terms"], cfg["name_aliases"]
    gu_names = list(config["busan_sigungu"].values())
    out_dir = resolve_path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    stores_raw = read_csv(resolve_path(cfg["store_path"]))
    merchants_raw = read_csv(resolve_path(cfg["merchant_path"]))
    emap = read_csv(resolve_path(cfg["exclusion_map"]))
    rules = read_csv(resolve_path(cfg["exclusion_keywords"]))

    files = pd.concat([inventory(stores_raw, "상가정보", cfg["store_reference"]),
                       inventory(merchants_raw, "가맹점", cfg["merchant_reference"])])
    files.to_csv(out_dir / "inventory.csv", index=False, encoding="utf-8-sig")

    stores = store_keys(classify_stores(stores_raw, emap, rules, legal, aliases), legal, aliases)
    merchants = prepare_merchants(merchants_raw, legal, aliases, gu_names)
    result, matched_ids = match(stores, merchants, cfg["name_similarity_min"], cfg["name_contain_min_len"])

    m_cols = merchants.set_index("merchant_id")[["name", "addr"]].rename(columns={"name": "가맹점_상호", "addr": "가맹점_주소"})
    full = stores.merge(result, on="상가업소번호").merge(m_cols, left_on="가맹점id", right_index=True, how="left")
    keep = ["상가업소번호", "상호명", "지점명", "도로명주소", "시군구명", "상권업종대분류명", "상권업종소분류명",
            "최종판정", "최종애매유형", "적용규칙", "매칭상태", "매칭규칙", "상호유사도", "가맹점_상호", "가맹점_주소"]
    full[keep].to_csv(out_dir / "match_result.csv", index=False, encoding="utf-8-sig")

    scenarios = cfg["scenarios"]
    summary = pd.concat([summarize(full, s) for s in scenarios]).drop(columns="구분")
    ends = [scenarios[0], scenarios[-1]]
    by_gu = pd.concat([summarize(full, s, "시군구명") for s in ends])
    by_cat = pd.concat([summarize(full, s, "상권업종대분류명") for s in ends])
    for name, table in (("summary_scenarios", summary), ("by_gu", by_gu), ("by_category", by_cat)):
        table.to_csv(out_dir / f"{name}.csv", index=False, encoding="utf-8-sig")

    seed, n = cfg["seed"], cfg["manual_check_n"]
    check = pd.concat([sample_check(full, STATUS_NONE, n, seed), sample_check(full, STATUS_FUZZY, n, seed)])
    check[keep].assign(앱확인결과="").to_csv(out_dir / "manual_check.csv", index=False, encoding="utf-8-sig")

    verdicts = (full.groupby(["최종판정", "최종애매유형"]).size().reset_index(name="가게수")
                .replace({"최종애매유형": {"": "-"}}))
    match_counts = full.groupby(["최종판정", "매칭상태"]).size().unstack(fill_value=0).reset_index()
    rule_counts = full.loc[full["매칭규칙"] != ""].groupby("매칭규칙").size().reset_index(name="가게수")
    write_report(out_dir / "report.md", {
        "store_ref": cfg["store_reference"], "merchant_ref": cfg["merchant_reference"],
        "summary": summary, "files": files.groupby(["파일", "기준시점"])["행수"].first().reset_index()
                                           .assign(컬럼수=files.groupby("파일").size().values),
        "merchant_rows": len(merchants_raw), "merchant_dedup": len(merchants),
        "merchant_prev": cfg["merchant_previous_count"], "merchant_unparsed": int(merchants["addr_key"].isna().sum()),
        "verdicts": verdicts, "match_counts": match_counts, "rule_counts": rule_counts, "by_gu": by_gu, "by_cat": by_cat,
        "merchant_unmatched": len(merchants) - len(matched_ids),
        "sim_min": cfg["name_similarity_min"], "contain_min": cfg["name_contain_min_len"],
        "out_dir": cfg["output_dir"], "n": n, "seed": seed,
    })
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
