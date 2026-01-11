import re
import unicodedata
from pathlib import Path
import pandas as pd

def normalize_text(s: str) -> str:
    s = str(s)
    s = unicodedata.normalize("NFKC", s)
    s = s.strip().lower()
    s = re.sub(r"[_\-]+", " ", s)                 # _ and - -> space
    s = re.sub(r"\s*\([^)]*\)\s*", " ", s)        # remove (actress) etc.
    s = re.sub(r"[^\w\s']+", " ", s)              # remove punctuation (keep apostrophes)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def load_master_names(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, keep_default_na=False)
    if len(df.columns) != 1:
        raise ValueError(f"Expected 1 column in master file, got {df.columns.tolist()}")
    col = df.columns[0]
    if col.lower().strip() != "name":
        df = df.rename(columns={col: "name"})
    return df[["name"]]

def load_new_names(path: Path) -> pd.DataFrame:
    # Many scraped lists have NO header → force header=None
    df = pd.read_csv(path, header=None, keep_default_na=False)
    if df.shape[1] != 1:
        raise ValueError(f"Expected 1 column in new file, got {df.shape[1]}")
    return df.rename(columns={0: "name"})[["name"]]

def merge_names(master_csv: str, new_csv: str, out_master_csv: str, out_new_only_csv: str):
    master_path = Path(master_csv)
    new_path = Path(new_csv)

    df_master = load_master_names(master_path)
    df_new = load_new_names(new_path)

    # clean empties
    df_master["name"] = df_master["name"].astype(str).str.strip()
    df_new["name"] = df_new["name"].astype(str).str.strip()
    df_master = df_master[df_master["name"] != ""].copy()
    df_new = df_new[df_new["name"] != ""].copy()

    df_master["name_norm"] = df_master["name"].map(normalize_text)
    df_new["name_norm"] = df_new["name"].map(normalize_text)

    # dedupe within each
    df_master = df_master.drop_duplicates(subset=["name_norm"])
    df_new = df_new.drop_duplicates(subset=["name_norm"])

    existing = set(df_master["name_norm"])
    df_new_only = df_new[~df_new["name_norm"].isin(existing)].copy()

    df_merged = pd.concat([df_master, df_new_only], ignore_index=True)
    df_merged = df_merged.drop_duplicates(subset=["name_norm"]).sort_values("name_norm").reset_index(drop=True)

    df_merged[["name"]].to_csv(out_master_csv, index=False, encoding="utf-8")
    df_new_only[["name"]].to_csv(out_new_only_csv, index=False, encoding="utf-8")

    print(f"Master original rows: {len(df_master)}")
    print(f"New list rows (unique after norm): {len(df_new)}")
    print(f"New names added: {len(df_new_only)}")
    print(f"Merged total unique: {len(df_merged)}")
    print(f"✅ Updated master saved to: {out_master_csv}")
    print(f"✅ New-only saved to: {out_new_only_csv}")

if __name__ == "__main__":
    merge_names(
        master_csv="data/names.csv",#path to existing master names CSV
        new_csv="data/nameonly.csv",
        out_master_csv="data/names_updated.csv",
        out_new_only_csv="data/names_new_only.csv",
    )
