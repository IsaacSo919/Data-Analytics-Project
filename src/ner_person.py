# src/ner_person.py
from __future__ import annotations
from typing import List, Set, Optional

class PersonNER:
    """
    Extract PERSON spans using a Transformers NER model.
    Optional: if transformers/torch not available, it returns empty list.
    """
    def __init__(self, model_name: str = "dslim/bert-base-NER"):#default model is dslim/bert-base-NER, it returns false if the model is not found
        self.model_name = model_name
        self._pipe = None
        # _pipe 是「快取」：之後會放 transformers pipeline 物件。

    def _lazy_load(self):
        # 這是 private-ish function（前面加底線的慣例）：用 lazy loading（用到才載入）。
        # this is a private-ish function (convention of leading underscore): use lazy loading (load only when needed).
        # 目的：避免一建立 PersonNER 就立刻下載模型、載入 torch，省時間也省資源。
        # Purpose: Avoid downloading the model and loading torch immediately when creating PersonNER, saving time and resources.
        if self._pipe is not None:
            # 如果 _pipe 不是 None，代表「已經嘗試載入過了」（成功或失敗都算）。
            return
        # 直接 return，不要重複載入（也避免一直重試下載模型）。
        try:
            from transformers import pipeline
            # aggregation_strategy merges subword pieces into full spans
            self._pipe = pipeline(
                "token-classification",# 建立 transformers pipeline，任務是 "token-classification"（NER 常用）。
                model=self.model_name,
                aggregation_strategy="simple",
            )
        except Exception:
            # transformers/torch missing or model download failed
            self._pipe = False

    def extract_person_candidates(self, text: str) -> List[str]:
        # 主要的 public function：輸入 text（str），輸出 List[str]（一堆名字）。
        # Main public function: input text (str), output List[str] (a list of names). 
        self._lazy_load()
        if not self._pipe:
            return []

        spans = self._pipe(text)
        out: Set[str] = set()
        for s in spans:
            # keys depend on pipeline version; handle both
            label = (s.get("entity_group") or s.get("entity") or "").upper()
            word = (s.get("word") or s.get("text") or "").strip()

            # CoNLL-style models use PER
            if label in ("PER", "PERSON") and word:
                out.add(word)

        return list(out)
