from pathlib import Path
import csv
import re

def is_name_start(tok: str) -> bool:
    # start token patterns
    if not tok:
        return False
    if tok.isupper() and len(tok) >= 2:
        return True
    if re.match(r"^[A-Z][a-z]+$", tok):  # TitleCase word
        return True
    if re.match(r"^[A-Za-z]+[0-9]+$", tok):  # username-like
        return True
    if re.match(r"^[A-Za-z]+[_-][A-Za-z0-9]+$", tok):  # underscore/hyphen
        return True
    if tok.startswith("The") and len(tok) > 3:  # TheIRISHbull-ish (rough)
        return True
    return False

def is_name_token(tok: str) -> bool:
    # tokens that can belong in a name
    if not tok:
        return False
    if re.match(r"^[A-Za-z]+$", tok):
        return True
    if re.match(r"^[A-Za-z]+[0-9]+$", tok):
        return True
    if re.match(r"^[A-Za-z]+[_-][A-Za-z0-9]+$", tok):
        return True
    return False

def split_blob_to_names(text: str) -> list[str]:
    # remove weird punctuation at the end of blob
    text = text.replace("\n", " ").strip()
    tokens = [t for t in re.split(r"\s+", text) if t]

    names = []
    cur = []

    def flush():
        nonlocal cur
        if cur:
            names.append(" ".join(cur))
            cur = []

    i = 0
    while i < len(tokens):
        tok = tokens[i]

        # If token is not "name-like", treat it as boundary (flush)
        if not is_name_token(tok):
            flush()
            i += 1
            continue

        if not cur:
            # start new name only if this looks like a start
            if is_name_start(tok) or re.match(r"^[a-z]", tok) is None:
                cur = [tok]
            else:
                cur = [tok]  # fallback, keep it anyway
            i += 1
            continue

        # If current has some tokens and we see a strong new-start signal,
        # decide whether to flush current and start a new name.
        if is_name_start(tok) and len(cur) >= 2:
            flush()
            cur = [tok]
            i += 1
            continue

        # Otherwise keep accumulating, but cap length
        cur.append(tok)
        if len(cur) >= 4:
            flush()
        i += 1

    flush()

    # Clean + dedupe (case-insensitive)
    out = []
    seen = set()
    for n in names:
        n = n.strip(" .,'\"")
        if not n:
            continue
        key = n.lower()
        if key not in seen:
            seen.add(key)
            out.append(n)
    return out

def main():
    inp = Path("data/names_blob.txt")
    outp = Path("data/names.csv")

    blob = inp.read_text(encoding="utf-8", errors="ignore")
    names = split_blob_to_names(blob)

    outp.parent.mkdir(parents=True, exist_ok=True)
    with outp.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name"])
        for n in names:
            w.writerow([n])

    print(f"✅ Wrote {outp} with {len(names)} rows.")
    print("First 20:")
    for x in names[:20]:
        print(" -", x)

if __name__ == "__main__":
    main()
