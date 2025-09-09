# Data Analysis Findings (2023-2024)

>This summary is based on the visualizations generated in the `charts/` directory. Each chart is referenced and described below.

---

## 1. User Type Distribution

**Chart:** ![User Type Distribution](charts/user_type_distribution.png)

- A significant amount of free trial users implies interest in membership
- Majority are regular users, with a notable premium segment.

---

## 2. Monthly Revenue Trends

**Chart:** ![Monthly Revenue](charts/monthly_revenue.png)

- Erratic first year, with a spike in September followed by a steep drop off that rebounds for the holidays
- Second year more stable, with a strong Q1, followed by a drop and then steady growth through the rest of the year

---

## 3. Category Revenue Analysis

**Chart:** ![Category Revenue](charts/category_revenue.png)

- Furniture makes up most of revenue share
- Relatively even revenue distribution among the top categories

---

## 4. Category Monthly Heatmap

**Chart:** ![Category Monthly Heatmap](charts/category_monthly_heatmap.png)

- Most categories follow a similar trend
- Furniture sales are worst in the summer

---

## 5. Spending Distribution (All Users)

**Chart:** ![Total Spending Distribution](charts/total_spending_distribution.png)

- Most users spend modest amounts; a few spend much more.
- Long-tail effect is evident.

---

## 6. Spending Distribution by Percentile

**Charts:**
- ![Bottom 90%](charts/bottom90_spending_distribution.png)
- ![Bottom 95%](charts/bottom95_spending_distribution.png)
- ![Bottom 97%](charts/bottom97_spending_distribution.png)
- ![Bottom 99%](charts/bottom99_spending_distribution.png)

- These charts exclude the top spenders
- The spending is concentrated in purchases under $500

---

For further details, see the full analysis in `data_analysis.py` and the above charts in the `charts/` directory.
