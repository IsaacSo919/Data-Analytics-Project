import argparse
from pathlib import Path

from src.data_ingestion import ingest_queries
from src.name_matching import load_names_csv, build_automaton, extract_names_from_query_norm, build_name_index
from src.ner_person import PersonNER

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--names", required=True)
    p.add_argument("--output", required=True)
    
    p.add_argument("--use-ner", action="store_true", help="Use Transformers NER as fallback when dictionary finds nothing")
    p.add_argument("--ner-model", default="dslim/bert-base-NER")
    p.add_argument("--min-count-for-ner", type=int, default=1000, help="Only run NER fallback on high-count queries")
    
    args = p.parse_args()

    df = ingest_queries(args.input, aggregate=False)

    names_norm = load_names_csv(args.names)
    names_set, canon_map = build_name_index(names_norm)
    A = build_automaton(names_norm)
    
    ner = PersonNER(args.ner_model) if args.use_ner else None
        
    names_out = []
    for q_raw, q_norm, cnt in zip(df["query"], df["query_norm"], df["count"]):
        # 1) Fast exact dictionary match first
        found = extract_names_from_query_norm(q_norm, A)

        # 2) NER fallback only if enabled and nothing found and query is high-value
        if (not found) and ner and cnt >= args.min_count_for_ner:
            cands = ner.extract_person_candidates(q_raw)
            verified = []
            for c in cands:
                v = verify_pornstar(c, names_set, canon_map)
                if v:
                    verified.append(v)
            # deterministic output
            found = ",".join(dict.fromkeys(verified))
        names_out.append(found)
    df["names"] = names_out
    df.to_csv(args.output, index=False, encoding="utf-8")
    print(f"✅ Saved to {args.output}")
    print(df[["query", "count", "names"]].head(10).to_string(index=False))    
    df["names"] = df["query_norm"].apply(lambda q: extract_names_from_query_norm(q, A))

    # Optional: if you don’t want query_norm in final output
    # df = df.drop(columns=["query_norm"])

    df.to_csv(args.output, index=False, encoding="utf-8")
    print(f"✅ Wrote: {args.output}")
    print(df[["query", "count", "names"]].head(10).to_string(index=False))

if __name__ == "__main__":
    main()
