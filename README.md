# TNAflix

Minimal workspace scaffold for the TNAflix project.

Quick start

- Python:
  - Create a virtual environment: `python -m venv .venv`
  - Activate: `.\.venv\Scripts\Activate.ps1`
  - Install deps: `pip install -r requirements.txt`
  - Run: `python -m src.main`

- Node (optional):
  - Install deps: `npm install`
  - Run: `npm start`

Project layout

- `src/` — source files (Python/Node examples)
- `tests/` — test files (pytest)
- `requirements.txt` — Python dependencies
- `package.json` — Node metadata (optional)

Command to run NER model only:
"""
The limit is changable, it's limiting how many data to be processed.
"""
python run_ner_only.py --input sample_train_unseen_20250105.csv --output ner_2k.csv --titlecase --limit 50000
