# Mini E‑Commerce Analytics Project

## Summary
This is a 2–3 day learning project for a 3‑person junior team. You will:
1. Load synthetic e‑commerce CSV data (users + purchases) into PostgreSQL.
2. Answer 5 business questions and produce 5 charts.
3. Build ONE simple machine learning model (classification, next category, or clustering).
4. Document findings and model results.

Focus on finishing a working pipeline over perfection.

## Key Docs
- Project Specification: [PROJECT_SPECIFICATION.md](PROJECT_SPECIFICATION.md)
- ML Guide (model options & code): [ML_GUIDE.md](ML_GUIDE.md)

## Core Files
```
PROJECT_SPECIFICATION.md  # Full scoped plan
ML_GUIDE.md               # Copy/paste ML examples
import_data.py            # CSV → PostgreSQL loader
data_analysis.py          # Queries + charts
ml_model.py               # Your chosen model
FINDINGS.md               # Analysis answers
MODEL_RESULTS.md          # ML performance summary
```

## Quick Start
1. Create PostgreSQL DB named `ecommerce`.
2. Run `database_setup.sql` to create tables.
3. Put monthly CSVs in `data/user_data/` and `data/purchase_data/`.
4. Run `import_data.py` to load data.
5. Run `data_analysis.py` to generate charts (ensure a `charts/` folder exists).
6. Pick ONE model in `ML_GUIDE.md` and implement it in `ml_model.py`.
7. Fill in `FINDINGS.md` and `MODEL_RESULTS.md`.

## Success Criteria (Abbreviated)
- Tables populated (non‑empty).
- 5 charts saved.
- 1 model trains & evaluates (> random or sensible clusters).
- Docs clear enough to rerun in <30 minutes.

## Tips
- If stuck >20 min: simplify or ask.
- Use small slices of data while prototyping.
- Commit early, commit often.

## Row Counts
Users: 6528
Purchases: 10234

## Work Done
Austin: Data Analysis
- *Note: I used the same user when working in the vm as Hayden, and didn't think about it much. This resulted in me inadvertently using her github credentials to commit all my changes. Whoops*

Happy building!