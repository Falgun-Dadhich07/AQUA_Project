# ON NIFTY100 — Project Summary & How to Run

Summary
-	This workspace contains data and scripts used to prepare and explore NIFTY100 constituent data and preprocessed CSVs for neural-network experiments.
-	Top-level folders:
	- `data/` — contains `nifty100_constituents.csv` and `nifty100.py` (utility script for NIFTY100 data).
	- `NNet_data/` — many per-ticker CSVs (different preprocessing variants) used for neural-network training/evaluation.
	- `python/` — additional Python scripts and helpers (project-specific scripts may live here).
	- `r/` — R scripts (if present).

Quick prerequisites
-	Python 3.8+ (Windows): install from python.org or use an existing environment.
-	Recommended Python packages (install as needed): `pandas`, `numpy`. Other packages (e.g., `scikit-learn`, `tensorflow`, `torch`) may be required by specific scripts in `python/` — check those scripts for exact imports.

Setup (Windows PowerShell)
```
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install --upgrade pip
# If you have a requirements.txt, run:
# pip install -r requirements.txt
# Otherwise install minimal packages:
pip install pandas numpy
```

How to run
-	Inspect or run the small utility in `data/`:
```
python data/nifty100.py
```
-	Load CSVs from `NNet_data/` with `pandas.read_csv()` for analysis or model training.
-	Run other project scripts from the `python/` directory as needed, e.g.:
```
python python/your_script.py
```

Notes & tips
-	This README is intentionally concise — consult individual scripts for required dependencies and usage examples (look at the script header or top-level docstring).
-	No files were modified; this README was added only.

If you'd like, I can also:
-	Generate a `requirements.txt` by scanning imports, or
-	Add short usage examples per-script.
