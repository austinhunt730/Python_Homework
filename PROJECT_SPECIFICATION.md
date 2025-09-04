# 2-3 Day Mini Analytics Project

**Duration:** 1-3 days  
**Team:** 3 engineers  
**Goal:** Create a simple data pipeline (DB → Charts → Basic ML)

## TL;DR (What You're Building)
You'll create a PostgreSQL database with two tables, analyze e-commerce data with 5 basic charts, and build ONE simple ML model.

## "Done" Checklist
1. PostgreSQL running locally with `users` + `purchases` loaded.  
2. Analysis script produces required answers + 5 charts saved to `./charts/`.  
3. One ML model (classification, prediction OR clustering) trained & evaluated.  
4. Documentation: setup, findings, model results, next ideas.  
5. All code committed with clear filenames.

## Success Criteria (Keep It Simple!)
- Data load completes with realistic row counts (no empty tables).  
- 5 business questions answered in `FINDINGS.md`.  
- Model runs end‑to‑end and beats random (classification > 50% accuracy OR clustering silhouette > 0.30).  
- Clear README so another dev can reproduce in <30 min.  
- No blocking errors when re-running scripts from clean clone.

---

## Technical Requirements

### Database
- PostgreSQL (local installation).  
- Two simple tables (users, purchases).  
- No complex indexing needed.

### Tools
- Python 3.8+ with: pandas, psycopg2-binary, matplotlib, scikit-learn
- Git (simple workflow - commit directly to main)
- Markdown files for documentation

### Data (Already Provided)
**User CSVs** (`user_data_YYYY_MM.csv`):
- Key fields: email, user_type, total_spent, purchase_count, device info

**Purchase CSVs** (`purchase_data_YYYY_MM.csv`):
- Key fields: transaction_id, user_email, product_name, category, price, date

---

## Plan Outline

### Database Setup
| Task | Output |
|------|--------|
| Create tables (`users`, `purchases`) | `database_setup.sql` |
| Load all monthly CSVs | `import_data.py` |
| Quick sanity counts | Row counts logged / noted in README |
| Capture issues (nulls, odd formats) | Section in `FINDINGS.md` |

Minimum schema (keep it simple):
```
users(email PK, first_name, last_name, user_type, total_spent, purchase_count, last_device)
purchases(transaction_id PK, user_email FK→users.email, product_name, product_category, total_price, purchase_date)
```

### Data Analysis
Answer these 5 questions (use SQL or pandas):
1. User type counts.  
2. Top 10 spenders.  
3. Device / browser distribution.  
4. Top 5 categories by revenue.  
5. Monthly revenue trend (2023–2024).  

Make these 5 charts (save as PNG files):
1. Bar chart: Revenue by category
2. Line chart: Monthly revenue (2023-2024)
3. Pie chart: User types (%)
4. Histogram: Total spending distribution
5. Any chart of your choice

Deliverables: `data_analysis.py`, `charts/`, findings section in `FINDINGS.md`.

### Machine Learning
Pick just ONE of these options:
- **Option A:** Predict user type (premium vs regular)
- **Option B:** Predict next purchase category (top 5 categories only)
- **Option C:** Cluster users by spending patterns (3 clusters)

Follow ML_GUIDE.md which has copy-paste code examples.
Success = any working model with >50% accuracy or sensible clusters.
Document what you learned in MODEL_RESULTS.md.

### If You Finish Early
- Add README.md with setup steps
- Try a different ML model type
- Add a "Future Ideas" section

---

## Team Tips
Suggested roles: (Decide how you want to divide the work amongst the team.)
- Person 1: Database setup & data import
- Person 2: Data analysis & charts
- Person 3: ML model & documentation

Git: Commit often with clear messages. Help each other when stuck.

---

## Helpful Constraints

### Keep It Very Simple
- Working code > elegant code
- If stuck >20 minutes, ask for help
- Focus on main requirements only

### Data Tips
- Use only the fields you need
- Ignore complex JSON fields
- Skip data cleaning beyond basics
- It's OK to filter out problematic rows

### It Doesn't Need To Be Fast
- Speed doesn't matter for this project
- Use the simplest approach that works
- Small sample of data is fine for testing

---

## Minimum Requirements

### Code
- Clear variable names
- Basic error handling for file/DB operations
- Comments for complex parts only
- Use random_state=42 for ML reproducibility

### Documentation
- Short README with setup steps
- FINDINGS.md with analysis results
- MODEL_RESULTS.md with ML outcomes

### Files Needed
```
project_root/
├── README.md                    # Main setup and overview
├── requirements.txt             # Python packages needed
├── database_setup.sql           # Database table creation
├── import_data.py               # Import CSV data to database
├── data_analysis.py             # Analysis and charts
├── ml_model.py                  # Machine learning model
├── FINDINGS.md                  # Data analysis results
├── MODEL_RESULTS.md             # ML model performance
├── ML_GUIDE.md                  # Step-by-step ML instructions
└── data/                        # Data folder
    user_data                    # User CSV files
    purchase_data                # Purchase CSV files
```

**What Each File Does:**
- `database_setup.sql`: Table creation
- `import_data.py`: Load CSVs to database
- `data_analysis.py`: Generate answers & charts
- `ml_model.py`: Train & evaluate ONE model
- `FINDINGS.md`: Your analysis answers
- `MODEL_RESULTS.md`: ML performance summary

---

## Final Checklist

✅ Tables created & populated  
✅ 5 charts generated & saved  
✅ 1 ML model working (any accuracy > random)  
✅ Findings documented  
✅ All team members contributed code

### You'll Learn/Reinforce
- PostgreSQL basics
- Data analysis with pandas
- Simple visualization
- ML model training basics
- Git collaboration

### Stretch Goals
- More insights in FINDINGS.md
- Better ML model accuracy
- "Next Steps" ideas section

---

## Quick Start (You choose how to accomplish this task)

1. **PostgreSQL Setup**
   - Install from postgresql.org
   - Create database: 'ecommerce'
   - Remember your password!

2. **Install Python Packages**
   ```bash
   pip install pandas psycopg2-binary matplotlib seaborn scikit-learn
   ```

3. **Data Files**
   - Unzip data files into `data/user_data/` and `data/purchase_data/`
   - Each folder has 24 monthly CSV files

4. **ML Quick Start**
   - Follow `ML_GUIDE.md` - it has complete copy-paste examples
   - Pick just ONE model type

### Database Code Sample
```sql
-- Sample table creation (expand on this)
CREATE TABLE users (
    email VARCHAR PRIMARY KEY,
    user_type VARCHAR,
    total_spent DECIMAL,
    purchase_count INTEGER
);
```

```python
# Database connection
conn = psycopg2.connect(
    host="localhost",
    database="ecommerce", 
    user="postgres",
    password="your_password"
)

# Read CSV
df = pd.read_csv('data/user_data/user_data_2023_01.csv')

# Save chart
plt.savefig('charts/revenue_by_category.png', dpi=120)
```

### Common Problems
- DB won't connect? Check service is running & password
- Charts not showing? Create 'charts/' folder first
- ML accuracy low? Try fewer features or simpler model
- Data issues? It's OK to filter out problematic rows

---

## Teamwork Tips
- Communicate what you're working on
- Commit code at least once per day
- Ask for help early if stuck
- Working code > perfect code

## Helpful SQL Samples
```sql
-- Count users by type
SELECT user_type, COUNT(*) FROM users GROUP BY user_type;

-- Top spending customers  
SELECT email, total_spent FROM users ORDER BY total_spent DESC LIMIT 10;

-- Monthly sales
SELECT EXTRACT(month FROM purchase_date) as month, SUM(total_price) 
FROM purchases GROUP BY month ORDER BY month;
```

### Remember
- The goal is LEARNING, not perfection
- Keep it small and achievable
- Focus on completing each step, not optimizing
- Help each other when stuck
