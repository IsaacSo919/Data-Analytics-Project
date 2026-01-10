from __future__ import annotations
from typing import List
import spacy

_nlp = None

def get_nlp(model: str = "en_core_web_sm"):
    global _nlp
    if _nlp is None:
        # keep only what's needed for NER (faster)
        _nlp = spacy.load(model, disable=["parser", "tagger", "lemmatizer", "attribute_ruler"])
    return _nlp

def extract_persons_batch(
    texts: List[str],
    model: str = "en_core_web_sm",
    batch_size: int = 256,
    n_process: int = 1,
    titlecase: bool = False,
) -> List[str]:
    """
    Returns one CSV string per input text, e.g. "Mia Khalifa,Johnny Sins"
    """
    nlp = get_nlp(model)

    if titlecase:
        texts = [t.title() for t in texts]

    out = []
    # nlp.pipe = batch processing (way faster than looping nlp(text) one by one)
    for doc in nlp.pipe(texts, batch_size=batch_size, n_process=n_process):
        seen = set()
        persons = []
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                key = ent.text.lower()
                if key not in seen:
                    seen.add(key)
                    persons.append(ent.text)
        out.append(",".join(persons))
    return out
