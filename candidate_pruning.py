# candidate_pruning.py
"""
Prune nearby store candidates so that ChatGPT only sees the k most plausible
matches for every target store.

USAGE
-----
import pandas as pd
from candidate_pruning import prune_candidates

df = pd.read_csv("stores.csv")                    # or however you load it
df_pruned = prune_candidates(df, k=3)             # adds a 'topk_nearby' col
df_pruned.to_parquet("stores_pruned.parquet")     # feed this to the GPT step
"""

from __future__ import annotations
import re, json, unicodedata
from typing import List, Dict
import pandas as pd
from rapidfuzz import fuzz
from pypinyin import lazy_pinyin

# --------------------------------------------------------------------------
# 1. Normalisation helpers
# --------------------------------------------------------------------------

_punct_re = re.compile(r"[\s\p{P}\p{S}]+", re.U)

def normalise(text: str) -> str:
    """Unicode NFKC fold, drop punctuation, lower-case, trim."""
    if not isinstance(text, str):                              # safety
        return ""
    t = unicodedata.normalize("NFKC", text)
    t = _punct_re.sub("", t)                                   # remove punctuation/space
    return t.lower().strip()

def han2pinyin(text: str) -> str:
    """Convert Chinese to continuous pinyin string (no tones/spaces)."""
    return "".join(lazy_pinyin(text, errors="ignore"))

# --------------------------------------------------------------------------
# 2. Heuristic score (0-100)
# --------------------------------------------------------------------------

def heuristic_score(target: str, cand: str, brand_dict: Dict[str, List[str]]) -> int:
    t_norm, c_norm = normalise(target), normalise(cand)
    if not t_norm or not c_norm:
        return 0

    # 2.1 Exact match (fast-exit)
    if t_norm == c_norm:
        return 100

    score = 0

    # 2.2 Brand synonym hit
    for brand, aliases in brand_dict.items():
        if brand in target or any(a in target for a in aliases):
            if brand in cand or any(a in cand for a in aliases):
                score += 20
                break

    # 2.3 Substring / alias overlap
    if t_norm in c_norm or c_norm in t_norm:
        score += 15

    # 2.4 Levenshtein ratio (main signal)
    score += int(25 * fuzz.ratio(t_norm, c_norm) / 100)

    # 2.5 Pinyin similarity
    p_t, p_c = han2pinyin(target), han2pinyin(cand)
    if p_t and p_c:
        score += int(15 * fuzz.ratio(p_t, p_c) / 100)

    # cap at 95, leaving 100 for exact matches
    return min(score, 95)

# --------------------------------------------------------------------------
# 3. Prune candidates
# --------------------------------------------------------------------------

def prune_candidates(df: pd.DataFrame,
                     k: int = 3,
                     brand_dict: Dict[str, List[str]] | None = None) -> pd.DataFrame:
    """
    Append a column 'topk_nearby' containing the k best store names
    (ranked by heuristic_score). Input df must have:
        - 'odp_store' (target store name, str)
        - 'nearby_store_names' (list/str-encoded list of names)
    """

    if brand_dict is None:                                     # minimal default
        brand_dict = {
            "肯德基": ["kfc", "k.f.c."],
            "麦当劳": ["mcd", "mcdonald", "麦麦"],
            # add more global / local chains if you like
        }

    def _prune_row(row):
        target = row["odp_store"]
        # handle both Python list and JSON-stringified list
        cands = row["nearby_store_names"]
        if isinstance(cands, str):
            try:
                cands = json.loads(cands)
            except json.JSONDecodeError:
                cands = []
        if not isinstance(cands, list):
            cands = []

        scored = [
            (cand, heuristic_score(target, cand, brand_dict))
            for cand in cands
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [c for c, _ in scored[:k]]

    # we keep the original structure untouched and add a new column
    df = df.copy()
    df["topk_nearby"] = df.apply(_prune_row, axis=1)
    return df
