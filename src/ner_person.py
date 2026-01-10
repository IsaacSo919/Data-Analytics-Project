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

        spans = self._pipe(text)# 跑 NER：pipeline(text) 會回傳一個 list of dicts（每個 dict 是一個 span 的資訊）
        #常見 keys：entity_group/entity、word/text、score、start、end 等（版本不同會有差）。
        out: Set[str] = set()
        # 建一個 set 來去重（deduplicate）。
        # 用 type hint 說 out 是 Set[str]。
        for s in spans:
            # keys depend on pipeline version; handle both
            label = (s.get("entity_group") or s.get("entity") or "").upper()
            word = (s.get("word") or s.get("text") or "").strip()
            # 有些版本用 "entity_group"（像 "PER"），有些用 "entity"（像 "B-PER"/"I-PER"）。
            # s.get(...) 拿不到就回 None；用 or 串起來做 fallback。
            # 最後 .upper() 統一成大寫，避免 "per"/"PER" 比對失敗。
            # CoNLL-style models use PER
            if label in ("PER", "PERSON") and word:
                ''' 過濾條件：
                # - label 必須是 PER 或 PERSON（人名）
                # - 而且 word 不能是空字串
                #filter conditions:
                - label must be PER or PERSON (person name)  
                - and word cannot be an empty string
                '''
                out.add(word)
                # add 到 set：自動去重。

        return list(out)
    # 把 set 轉回 list 回傳。
    # 注意：set 轉 list 的順序「不保證固定」，所以每次跑出來順序可能不同。
