import argparse
from pathlib import Path
from src.normalisation import normalize_text
import pandas as pd

from src.data_ingestion import ingest_queries
from src.normalisation import normalize_text
from src.name_matching import load_names_csv, build_automaton, extract_names_from_query_norm

import spacy

def load_denylist(path: str) -> set[str]:
    p = Path(path)
    if not p.exists():
        return set()

    out = set()
    for line in p.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        out.add(normalize_text(s))
    return out

def spacy_person_candidates(text: str, nlp) -> list[str]:
    doc = nlp(text)
    out = []
    seen = set()

    for ent in doc.ents:
        if ent.label_ == "PERSON":
            raw = ent.text.strip()
            if not raw:
                continue
            key = raw.lower()
            if key not in seen:
                seen.add(key)
                out.append(raw)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--names", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--spacy-model", default="en_core_web_sm")
    p.add_argument("--denylist", default="data/denylist.txt")

    args = p.parse_args()

    # 1) Load data and denylist
    deny = load_denylist(args.denylist)
    df = ingest_queries(args.input, aggregate=False)
    if args.limit and args.limit > 0:
        df = df.head(args.limit).copy()

    # 2) Load pornstar names
    names_norm = load_names_csv(args.names)  # should return normalized list OR raw? (your current implementation)
    # IMPORTANT: ensure names_norm are normalized. If load_names_csv returns raw names, normalize them here:
    names_norm = [normalize_text(n) for n in names_norm]
    names_set = set(names_norm)

    # 3) Build fast matcher for known pornstars
    A = build_automaton(names_norm)

    # 4) Load spaCy NER
    nlp = spacy.load(args.spacy_model)

    known_names_out = []
    person_not_in_names_out = []

    for q_raw, q_norm in zip(df["query"], df["query_norm"]):
        # A) known pornstar extraction (dictionary-based)
        known = extract_names_from_query_norm(q_norm, A)
        known_names_out.append(known)

        # B) spaCy person candidates
        cands = spacy_person_candidates(str(q_raw), nlp)

        # C) keep only candidates NOT in names.csv (after normalization)
        unknown = []
        seen_u = set()
        for c in cands:
            cn = normalize_text(c)

            # 1) empty / 2) in denylist / 3) already known pornstar
            if (not cn) or (cn in deny) or (cn in names_set):
                continue
            if len(cn.split()) < 2:
                continue
            if len(cn) < 5:
                continue
            unknown.append(c)


        person_not_in_names_out.append(",".join(unknown))

    df["names"] = known_names_out
    df["person_not_in_names"] = person_not_in_names_out

    out_path = Path(args.output)
    df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"✅ Saved: {out_path}")
    print(df[["query", "count", "names", "person_not_in_names"]].head(15).to_string(index=False))


if __name__ == "__main__":
    main()
