import os, glob
from io import StringIO
import pandas as pd
import psycopg2

# ---- DB config via env vars ----
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_NAME = os.getenv("DB_NAME", "ecommerce")
DB_USER = os.getenv("DB_USER", "ecommerce_user")
DB_PASSWORD = os.getenv("DB_PASSWORD", "temp_pw")
DB_PORT = int(os.getenv("DB_PORT", "5432"))

conn = psycopg2.connect(host=DB_HOST, dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD, port=DB_PORT)

query = """
SELECT * FROM users;
"""

users_df = pd.read_sql_query(query, conn)
print("Users DataFrame:")
print(users_df.head())

## TODO Replace with read from database once all user data is imported


## QUESTIONS

# User type counts
# print(df['user_type'].value_counts())

# Top 10 spenders


# Device / browser distribution


# Top 5 categories by revnenue


# Monthly revenue trend (2023-2024)


# Make these 5 charts (save as PNG files):

# Bar chart: Revenue by category
# Line chart: Monthly revenue (2023-2024)
# Pie chart: User types (%)
# Histogram: Total spending distribution
# Any chart of your choice