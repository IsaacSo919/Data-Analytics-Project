import pandas as pd
from src.normalisation import normalize_text

REQUIRED_COLS = {"query", "count"}#define a dictionary to map column names
def ingest_queries(csv_path: str,aggregate: bool = False) -> pd.DataFrame:
    # 1) 讀 CSV：把 query/count 都先當字串讀進來（避免 pandas 自己亂猜型別）
    #read query/count as string to avoid pandas guessing types
    df = pd.read_csv(
        csv_path,
        dtype={"query": "string", "count": "string"},
        keep_default_na=False,   # missing values won't be converted to NaN
        on_bad_lines="skip",     # pandas will skip bad lines
    )
    # 2) 檢查必需欄位是否存在
    '''The following code checks if the required columns are present 
        in the dataframe.'''
    missing = REQUIRED_COLS - set(df.columns) #using set difference to find missing columns,這裡運用了數學中的 「集合減法」 (Set Difference) 概念,reference word file item 1
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Found: {list(df.columns)}")
    # 3) 清理 query：去前後空白，刪掉空 query
    df["query"] = df["query"].astype("string").str.strip()# 1.change type to string ,For example if "123"comes in, pd may view it as a num rather than string.strip() removes leading and trailing whitespace from each query string
    df = df[df["query"].notna() & (df["query"] != "")]#Boolean Indexing,notna means data is not NaN, so the whole sentense translates to "keep only rows where query is not NaN and query is not an empty string"
    # 4) normalized query
    df["query_norm"] = df["query"].apply(normalize_text)#applied the normalize_text function to each entry in the "query" column and store the results in a new column called "query_norm"

    # 5) 清理 count：轉數字，轉不了的變 NaN，再補 0，最後轉成 int
    '''part five is to clean the count column by converting it to numeric, 
        replacing non-convertible values with NaN, filling NaN with 0, 
        and finally converting to int64.'''
    df["count"] = pd.to_numeric(df["count"], errors="coerce").fillna(0).astype("int64")
    df = df[df["count"] >= 0]  # count should be non-negative
    # Drop index-like columns (optional but nice)
    df = df.loc[:, ~df.columns.str.match(r"^Unnamed")]
    # 6) （可選）合併重複 query：同一個 query 的 count 加總
    '''The sixth step is to aggregate duplicate queries by summing their counts.'''
    if aggregate:
        df = df.groupby(["query", "query_norm"], as_index=False)["count"].sum()
    # df = df.groupby(["query", "query_norm"], as_index=False)["count"].sum() # this would overwrite the query_norm, so we updated this to this version the output will be query, query_norm, count.
     # 7) quick check: print summary statistics and top queries.
    print(f"Rows: {len(df):,} | Unique queries: {df['query'].nunique():,}")
    print("Top 5 queries by count:")
    print(df.sort_values("count", ascending=False).head(5).to_string(index=False))

    return df