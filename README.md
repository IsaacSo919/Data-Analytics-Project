Can't wait to run this program?
Yea!
## Setup + Run (single supported method)

### 1) Create virtual environment (Python 3.12.3)
```powershell
py -3.12 -m venv .venv
.venv\Scripts\activate
python --version

2) Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

3) Run (the only command)
python submit.py --input sample_train_unseen_20250105.csv --names data\names.csv --output out_dict.csv --denylist data\denylist.txt

# TNAflix
A data engineering and NLP project for identifying **pornstar names** from large-scale, noisy search query data.

The project focuses on building a **scalable, testable pipeline** that combines rule-based matching and modern NLP techniques to extract full person names from organic user queries.

---
Project Layout

                    ┌──────────────────────────┐
                    │  Input CSV (100K queries)│
                    │  columns: query, count   │
                    └─────────────┬────────────┘
                                  │
                                  ▼
                      ┌───────────────────────┐
                      │ src/data_ingestion.py │
                      │ - read CSV            │
                      │ - validate columns    │
                      │ - clean count/query   │
                      │ - add query_norm      │
                      └─────────────┬─────────┘
                                    │
          ┌─────────────────────────┼─────────────────────────┐
          │                         │                         │
          ▼                         ▼                         ▼

┌───────────────────────┐  ┌──────────────────────────┐  ┌───────────────────────┐
│ Dictionary Matching    │  │ NER-Only Experiment       │  │ NER Audit for Expansion│
│ src/name_matching.py   │  │ run_ner_only.py           │  │ run_spacy_audit.py     │
│ + names.csv            │  │ (GPU batch)               │  │ (spaCy PERSON)         │
│ → output: names        │  │ → output: ner_persons     │  │ → output:              │
│ (known pornstars)      │  │ (no verification)         │  │   - names (known)      │
└─────────────┬─────────┘  └──────────────┬───────────┘  │   - person_not_in_names│
              │                            │              │     (unknown candidates)│
              ▼                            ▼              └───────────────┬─────────┘
   ┌──────────────────────┐     ┌──────────────────────┐                  │
   │ run_extract.py        │     │ output_ner_only.csv   │                  ▼
   │ produce final 100K    │     └──────────────────────┘       ┌────────────────────────┐
   │ output_100k.csv       │                                    │ Candidate Review Loop    │
   └──────────┬───────────┘                                    │ - count top candidates   │
              ▼                                                │ - human review (keep=1)  │
   ┌──────────────────────┐                                    │ - merge into names.csv   │
   │ output_100k.csv       │                                    └───────────────┬────────┘
   │ columns:              │                                                    ▼
   │ query,count,names     │                                      ┌────────────────────────┐
   └──────────────────────┘                                      │ scripts/merge_names.py │
                                                                  │ updates names.csv      │
                                                                  └────────────────────────┘


## Quick Start

### 1. Set up environment
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .\.venv312\Scripts\activate
pip install -r requirements.txt


2) Run unit tests

Run all tests:

python -m unittest discover -s tests -v


Run a single test module:

python -m unittest -v tests.test_ingestion_unittest
python -m unittest -v tests.test_normalisation_unittest
python -m unittest -v tests.test_name_matching_unittest
python -m unittest -v tests.test_ner_person_unittest

3) Run dictionary-based extraction (recommended baseline)

This uses the curated pornstar name list + exact matching.

python run_extract.py --input sample_train_unseen_20250105.csv --names data/names.csv --output output_100k.csv

4) Run hybrid extraction (dictionary + NER fallback)

This runs dictionary matching first, then uses Transformer NER only if:

dictionary finds nothing, and

the query count is high enough (to control compute)

python run_extract.py \
  --input sample_train_unseen_20250105.csv \
  --names data/names.csv \
  --output output_100k_hybrid.csv \
  --use-ner \
  --min-count-for-ner 1000

5) Run NER-only (experimental)

This extracts general PERSON entities (not pornstar-verified).

python run_ner_only.py \
  --input sample_train_unseen_20250105.csv \
  --output output_ner_only_50k.csv \
  --titlecase \
  --limit 50000

or 

python run_ner_only.py --input sample_train_unseen_20250105.csv --output ner_100k_gpu.csv --titlecase --limit 100000 --device 0 --batch-size 64

6) Performance experiments (NER batching)

If your NER runner supports batching:

python run_ner_only.py \
  --input sample_train_unseen_20250105.csv \
  --output output_ner_only_50k.csv \
  --titlecase \
  --limit 50000 \
  --batch-size 64

7) GPU check (optional)

Check whether PyTorch sees your GPU:

python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU')"


If CUDA is available, set device=0 inside the Transformers pipeline.


If you want, paste your current `run_extract.py` and `run_ner_only.py` argument list (`argparse` section). I can tailor the README commands to **exactly match** the flags you implemented so you don’t get “unrecognized arguments” later.
::contentReference[oaicite:0]{index=0}

8)If you scrape new names, formate them using the script/blob_to_names_csv.py
then run python scripts/merge_names.py.
Output are names new only and names updated.csv
Then move them back to names.csv

9)Combining all things together, Using ERV+database name matching
python run_spacy_audit.py --input sample_train_unseen_20250105.csv --names data\names.csv --output audit_2k.csv --limit 2000
python run_spacy_audit.py --input sample_train_unseen_20250105.csv --names data\names.csv --output audit_100k.csv.

10) Get those that is being searched a lot but not in the name.csv
python scripts\make_candidate_list.py --audit audit_100k.csv --out candidates_top500.csv --top 500

11) Run with denylist:
python run_spacy_audit.py --input sample_train_unseen_20250105.csv --names data\names.csv --output audit_100k.csv --denylist data\denylist.txt


