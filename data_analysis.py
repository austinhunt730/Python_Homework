import os, glob
from io import StringIO
import pandas as pd
from sqlalchemy import create_engine



# ---- DB config via env vars ----
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_NAME = os.getenv("DB_NAME", "ecommerce")
DB_USER = os.getenv("DB_USER", "ecommerce_user")
DB_PASSWORD = os.getenv("DB_PASSWORD", "temp_pw")
DB_PORT = int(os.getenv("DB_PORT", "5432"))

# Create SQLAlchemy engine
engine = create_engine(f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}")

query = """
SELECT * FROM users;
"""
users_df = pd.read_sql_query(query, engine)

query = """
SELECT * FROM purchases;
"""
purchases_df = pd.read_sql_query(query, engine)
print(f"Users table row count: {len(users_df)}")
print(f"Purchases table row count: {len(purchases_df)}")

## TODO Replace with read from database once all user data is imported


## QUESTIONS

# User type counts
user_type_counts = users_df['user_type'].value_counts()
print("\nUser Type Counts:")
print(user_type_counts)
print('\n')

# Top 10 spenders
top_spenders = purchases_df.groupby('user_email')['total_price'].sum().nlargest(10)
top_spenders_df = top_spenders.reset_index().merge(users_df[['email', 'first_name', 'last_name']], left_on='user_email', right_on='email', how='left')
print("\nTop 10 Spenders (with Names):")
print(top_spenders_df[['user_email', 'first_name', 'last_name', 'total_price']])
print('\n')

# Device / browser distribution
device_counts = users_df['last_device'].value_counts()
browser_counts = users_df['last_browser'].value_counts()
print("\nDevice Distribution:")
print(device_counts)
print("\nBrowser Distribution:")
print(browser_counts)

device_browser_counts = users_df.groupby(['last_device', 'last_browser']).size().reset_index(name='counts')
print("\nDevice / Browser Combined Distribution:")
print(device_browser_counts)
print('\n')

# Top 5 categories by revnenue
category_revenue = purchases_df[['product_category', 'total_price']].groupby('product_category').sum().nlargest(5, 'total_price').sort_values(by='total_price', ascending=False)
print("\nTop 5 Categories by Revenue:")
print(category_revenue)
print('\n')

# Monthly revenue trend (2023-2024)
monthly_revenue = purchases_df[['month', 'year', 'total_price']].groupby(['year', 'month']).sum().reset_index().sort_values(by=['year', 'month'])
monthly_revenue['change'] = monthly_revenue['total_price'].diff().fillna(0)
monthly_revenue['change'] = monthly_revenue['change'].apply(
    lambda x: f"+{x:.2f}" if x > 0 else (f"{x:.2f}" if x < 0 else "0.00")
)
print("\nMonthly Revenue Trend (2023-2024) with Change Indicator (amount):")
print(monthly_revenue)
print('\n')

# Make these 5 charts (save as PNG files):
import matplotlib.pyplot as plt
import seaborn as sns

# Bar chart: Revenue by category
category_revenue.reset_index().plot(kind='bar', x='product_category', y='total_price', legend=False)
plt.title('Top 5 Categories by Revenue')
plt.xlabel('Product Category')
plt.ylabel('Total Revenue (Millions USD)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('charts/category_revenue.png')
plt.clf()

# Line chart: Monthly revenue (2023-2024)
import matplotlib.dates as mdates

# ensure datetime x-axis and sorted order
monthly_revenue['month_year_dt'] = pd.to_datetime(
    monthly_revenue['year'].astype(int).astype(str) + '-' +
    monthly_revenue['month'].astype(int).astype(str) + '-01'
)
monthly_revenue = monthly_revenue.sort_values('month_year_dt')

fig, ax = plt.subplots(figsize=(12,6))
ax.plot(monthly_revenue['month_year_dt'], monthly_revenue['total_price'], marker='o', zorder=2)
ax.set_title('Monthly Revenue Trend (2023-2024)')
ax.set_xlabel('Month-Year')
ax.set_ylabel('Total Revenue (USD)')

# format x axis as YYYY-MM
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.xticks(rotation=45)

# shade quarters and label them
monthly_revenue['quarter'] = monthly_revenue['month_year_dt'].dt.to_period('Q')
quarters = monthly_revenue['quarter'].unique()
# higher-contrast alternating quarter backgrounds (soft blue and soft peach)
colors = ['#cce5ff', '#ffdfc4']
ymin, ymax = ax.get_ylim()
ytext = ymax * 0.95

for i, q in enumerate(quarters):
    q_start = q.start_time
    q_end = q.end_time + pd.Timedelta(days=1)  # make end inclusive
    ax.axvspan(q_start, q_end, color=colors[i % 2], alpha=0.4, zorder=0)
    mid = q_start + (q_end - q_start) / 2
    ax.text(mid, ytext, str(q), ha='center', va='top', fontsize=10, fontweight='bold', zorder=3)

plt.tight_layout()
plt.savefig('charts/monthly_revenue.png')
plt.clf()


# Pie chart: User types (%)
plt.figure(figsize=(8,8))
user_type_counts.plot(
    kind='pie',
    labels=[f"{idx} ({val/sum(user_type_counts)*100:.1f}%)" for idx, val in user_type_counts.items()],
    autopct=None,
    startangle=140,
    textprops={'fontsize': 16}
)
plt.title('User Type Distribution', fontsize=20)
plt.ylabel('')
plt.tight_layout()
plt.savefig('charts/user_type_distribution.png')
plt.clf()

# Histogram: Total spending distribution
plt.figure(figsize=(15,6))
sns.histplot(purchases_df['total_price'], bins=50, kde=True)
plt.title('Total Spending Distribution')
plt.xlabel('Total Price (USD)')
plt.ylabel('Number of Purchases')
plt.tight_layout()
plt.savefig('charts/total_spending_distribution.png')
plt.clf()

plt.figure(figsize=(15,6))
s = purchases_df['total_price'].dropna()
p99 = s[s <= s.quantile(0.99)]
sns.histplot(p99, bins=50, kde=True)
plt.title('Total Spending Distribution')
plt.xlabel('Total Price (USD)')
plt.ylabel('Number of Purchases')
plt.tight_layout()
plt.savefig('charts/bottom99_spending_distribution.png')
plt.clf()

plt.figure(figsize=(15,6))
s = purchases_df['total_price'].dropna()
p97 = s[s <= s.quantile(0.97)]
sns.histplot(p97, bins=50, kde=True)
plt.title('Total Spending Distribution')
plt.xlabel('Total Price (USD)')
plt.ylabel('Number of Purchases')
plt.tight_layout()
plt.savefig('charts/bottom97_spending_distribution.png')
plt.clf()

plt.figure(figsize=(15,6))
s = purchases_df['total_price'].dropna()
p95 = s[s <= s.quantile(0.95)]
sns.histplot(p95, bins=50, kde=True)
plt.title('Total Spending Distribution')
plt.xlabel('Total Price (USD)')
plt.ylabel('Number of Purchases')
plt.tight_layout()
plt.savefig('charts/bottom95_spending_distribution.png')
plt.clf()

plt.figure(figsize=(15,6))
s = purchases_df['total_price'].dropna()
p90 = s[s <= s.quantile(0.90)]
sns.histplot(p90, bins=50, kde=True)
plt.title('Total Spending Distribution')
plt.xlabel('Total Price (USD)')
plt.ylabel('Number of Purchases')
plt.tight_layout()
plt.savefig('charts/bottom90_spending_distribution.png')
plt.clf()

# Any chart of your choice
# Heatmap: Top categories by monthly revenue (category vs month-year)
pivot = purchases_df.groupby(['product_category', 'year', 'month'])['total_price'].sum().reset_index()
# build month-year for ordering
pivot['month_year_dt'] = pd.to_datetime(pivot['year'].astype(int).astype(str) + '-' + pivot['month'].astype(int).astype(str) + '-01')
pivot['month_year'] = pivot['month_year_dt'].dt.strftime('%Y-%m')

# pick top categories overall
top_categories = purchases_df.groupby('product_category')['total_price'].sum().nlargest(10).index.tolist()
pivot_top = pivot[pivot['product_category'].isin(top_categories)]

# pivot into matrix: rows=category, cols=month-year
pv = pivot_top.pivot_table(index='product_category', columns='month_year_dt', values='total_price', aggfunc='sum').fillna(0)

# sort columns chronologically and convert column labels to YYYY-MM
pv = pv.reindex(sorted(pv.columns), axis=1)
pv.columns = [c.strftime('%Y-%m') for c in pv.columns]

plt.figure(figsize=(14,8))
sns.heatmap(pv, cmap='YlGnBu', linewidths=0.5, linecolor='gray', cbar_kws={'label': 'Revenue (USD)'}, fmt='.0f')
plt.title('Monthly Revenue Heatmap for Top 10 Categories')
plt.xlabel('Month-Year')
plt.ylabel('Product Category')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('charts/category_monthly_heatmap.png')
plt.clf()
