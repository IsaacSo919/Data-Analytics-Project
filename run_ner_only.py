import argparse
from pathlib import Path
import pandas as pd
from tqdm import tqdm
tqdm.pandas()



def build_ner_pipeline(model_name: str):
    from transformers import pipeline
    # aggregation_strategy merges subword tokens into full spans
    return pipeline(
        "token-classification",
        model=model_name,
        aggregation_strategy="simple",
    )

def extract_persons(text: str, ner_pipe, titlecase: bool = False) -> str:
    if text is None:
        return ""
    s = str(text)

    # Optional trick: title-case can help models on lowercase queries
    # but it can also distort weird spellings. Use for testing only.
    if titlecase:
        s = s.title()

    spans = ner_pipe(s)
    out = []
    seen = set()

    for sp in spans:
        label = (sp.get("entity_group") or sp.get("entity") or "").upper()
        word = (sp.get("word") or sp.get("text") or "").strip()

        # CoNLL-style NER model uses PER
        if label in ("PER", "PERSON") and word:
            key = word.lower()
            if key not in seen:
                seen.add(key)
                out.append(word)

    return ",".join(out)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--model", default="dslim/bert-base-NER")
    p.add_argument("--titlecase", action="store_true", help="Title-case queries before NER (testing)")
    p.add_argument("--limit", type=int, default=0)

    args = p.parse_args()
    
    input_path = Path(args.input)
    output_path = Path(args.output)

    df = pd.read_csv(input_path, dtype={"query": "string"}, keep_default_na=False)
    if args.limit and args.limit > 0:
        df = df.head(args.limit)

    if "query" not in df.columns:
        raise ValueError(f"Input CSV must contain 'query'. Found: {list(df.columns)}")

    # Drop index-like columns if present
    df = df.loc[:, ~df.columns.str.match(r"^Unnamed")]

    ner_pipe = build_ner_pipeline(args.model)

    df["ner_persons"] = df["query"].progress_apply(
    lambda q: extract_persons(q, ner_pipe, titlecase=args.titlecase)
)

    df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"✅ Saved: {output_path}")
    print(df[["query", "ner_persons"]].head(10).to_string(index=False))

if __name__ == "__main__":
    main()
