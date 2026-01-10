import argparse
from pathlib import Path

from src.data_ingestion import ingest_queries
from src.name_matching import load_names_csv, build_automaton, extract_names_from_query_norm

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--names", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    df = ingest_queries(args.input, aggregate=False)

    names = load_names_csv(args.names)
    A = build_automaton(names)

    df["names"] = df["query_norm"].apply(lambda q: extract_names_from_query_norm(q, A))

    # Optional: if you don’t want query_norm in final output
    # df = df.drop(columns=["query_norm"])

    df.to_csv(args.output, index=False, encoding="utf-8")
    print(f"✅ Wrote: {args.output}")
    print(df[["query", "count", "names"]].head(10).to_string(index=False))

if __name__ == "__main__":
    main()
