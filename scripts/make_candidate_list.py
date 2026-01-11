import argparse
import pandas as pd
from pathlib import Path

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--audit", required=True, help="audit CSV produced by run_spacy_audit.py")
    p.add_argument("--out", required=True, help="output CSV of ranked candidates")
    p.add_argument("--top", type=int, default=500, help="top N candidates to export")
    args = p.parse_args()

    df = pd.read_csv(args.audit, keep_default_na=False)

    if "person_not_in_names" not in df.columns:
        raise ValueError("Missing column: person_not_in_names")

    # Split comma-separated values -> explode to one row per candidate
    s = (
        df["person_not_in_names"]
        .astype(str)
        .str.split(",")
        .explode()
        .astype(str)
        .str.strip()
    )

    # Remove empties
    s = s[s != ""]

    # Count frequency
    counts = s.value_counts().reset_index()
    counts.columns = ["candidate_name", "freq"]

    # Output top N
    out_df = counts.head(args.top)
    out_path = Path(args.out)
    out_df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"✅ Saved candidate list: {out_path}")
    print(out_df.head(30).to_string(index=False))

if __name__ == "__main__":
    main()
