from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple, Dict

import pandas as pd
import ahocorasick

from src.normalisation import normalize_text
@dataclass(frozen=True) #auto generate init, repr, eq, etc.
class Match:
    start: int
    end: int      # inclusive
    name: str     # normalised name

def load_names_csv(names_csv_path: str | Path) -> List[str]:   
    """
    Expect a CSV with a column named 'name'.
    Returns a list of normalised names (deduped).
    """
    df = pd.read_csv(names_csv_path, dtype={"name": "string"}, keep_default_na=False)
    if "name" not in df.columns:
        raise ValueError(f"names CSV must contain a 'name' column. Found: {list(df.columns)}")

    names = []
    seen = set()
    for raw in df["name"].tolist():
        n = normalize_text(raw)
        if n and n not in seen:
            seen.add(n)
            names.append(n)
    return names
def build_automaton(names_norm: Iterable[str]) -> ahocorasick.Automaton:
    """
    Build Aho-Corasick automaton mapping pattern -> pattern (normalised).
    Aho-Corasick is a string-matching algorithm that can match multiple patterns simultaneously in linear time.
    3. 圖解原理
    想像你要找 apple 和 apply：

    Trie 樹結構：它們共享前綴 a -> p -> p -> l。

    分支：到 l 之後，一條路通往 e，一條路通往 y。

    失敗指針：如果你在找某個字（例如 snapple）時，在 s 失敗了，它會立刻知道可以直接跳到 a 開始檢查有沒有 apple。
    """
    A = ahocorasick.Automaton()
    for name in names_norm:
        A.add_word(name, name)
    A.make_automaton()
    return A

def _is_token_boundary(s: str, start: int, end: int) -> bool:
    """
    s is already normalised: only [a-z0-9 ] with collapsed spaces.
    Token boundary means: left is start or space, right is end or space.
    Basically, this function is to avoid partial matches within words.
    For example, if we are looking for apple, pineapple should not count as a match.
    This is done by checking the characters immediately before and after the matched substring.
    """
    left_ok = (start == 0) or (s[start - 1] == " ")
    right_ok = (end == len(s) - 1) or (s[end + 1] == " ")
    return left_ok and right_ok


def find_matches(query_norm: str, automaton: ahocorasick.Automaton) -> List[Match]:
    """
    Return all token-boundary matches in the query (with spans).
    """
    q = "" if query_norm is None else str(query_norm)#checking for None input as above
    matches: List[Match] = [] #list to store matches

    for end_idx, name in automaton.iter(q):
        '''automaton.iter(q)：這是 Aho-Corasick 最強大的地方。它開始掃描字串 q。每當它發現一個關鍵字時，它會立刻告訴你兩件事：

            end_idx：這個關鍵字在什麼位置結束。

            name：找到的關鍵字名稱是什麼。
        '''
        start_idx = end_idx - len(name) + 1
        if start_idx < 0:
            continue
        if _is_token_boundary(q, start_idx, end_idx):
            matches.append(Match(start=start_idx, end=end_idx, name=name))

    return matches


def resolve_overlaps(matches: List[Match]) -> List[Match]:
    """
    Longest-span wins: keep longer matches first, discard overlaps.
    """
    # Sort by length desc, then start asc
    matches_sorted = sorted(matches, key=lambda m: (-(m.end - m.start + 1), m.start))

    kept: List[Match] = []
    for m in matches_sorted:
        overlap = any(not (m.end < k.start or m.start > k.end) for k in kept)
        if not overlap:
            kept.append(m)

    # Return in reading order
    kept.sort(key=lambda m: m.start)
    return kept


def matches_to_names(matches: List[Match]) -> str:
    """
    Deduplicate names while preserving order, return comma-separated.
    """
    ordered = []
    seen = set()
    for m in matches:
        if m.name not in seen:
            seen.add(m.name)
            ordered.append(m.name)
    return ",".join(ordered)


def extract_names_from_query_norm(query_norm: str, automaton: ahocorasick.Automaton) -> str:
    """
    Convenience wrapper: find -> resolve overlaps -> output string.
    """
    ms = find_matches(query_norm, automaton)
    ms = resolve_overlaps(ms)
    return matches_to_names(ms)
