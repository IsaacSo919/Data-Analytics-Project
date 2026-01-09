import re
import unicodedata

_space_re = re.compile(r"\s+") #referen: word file item 2

def normalize_text(s: str) -> str:
    s = "" if s is None else str(s) #Ternary Operator

    # Unicode normalize
    s = unicodedata.normalize("NFKC", s)

    # lowercase
    s = s.lower()

    # punctuation-like chars -> space
    s = re.sub(r"[^a-z0-9]+", " ", s)

    # collapse whitespace
    s = _space_re.sub(" ", s).strip()

    return s
