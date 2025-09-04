#!/usr/bin/env python3
import os, glob
import pandas as pd
import psycopg2
from io import StringIO

DB_HOST=os.getenv("DB_HOST","localhost")
DB_NAME=os.getenv("DB_NAME","ecommerce")
DB_USER=os.getenv("DB_USER","ecommerce_user")
DB_PASSWORD=os.getenv("DB_PASSWORD","temp_pw")
DB_PORT=int(os.getenv("DB_PORT","5432"))

def copy_df(conn, df, table):
    # Stream a CSV from memory into Postgres with COPY
    buf = StringIO()
    df.to_csv(buf, index=False, header=True)
    buf.seek(0)
    with conn.cursor() as cur:
        cur.copy_expert(f"COPY {table} FROM STDIN WITH CSV HEADER", buf)
    conn.commit()

def load_users(conn, pattern="data/user_data/*.csv"):
    files = sorted(glob.glob(pattern))
    if not files:
        print("No user CSVs found at", pattern); return
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    if "signup_date" in df.columns:
        df["signup_date"] = pd.to_datetime(df["signup_date"], errors="coerce").dt.date
    copy_df(conn, df, "users")
    print(f"Loaded users: {len(df):,}")

def load_purchases(conn, pattern="data/purchase_data/*.csv"):
    files = sorted(glob.glob(pattern))
    if not files:
        print("No purchase CSVs found at", pattern); return
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    if "purchased_at" in df.columns:
        df["purchased_at"] = pd.to_datetime(df["purchased_at"], errors="coerce")
    copy_df(conn, df, "purchases")
    print(f"Loaded purchases: {len(df):,}")

if __name__ == "__main__":
    conn = psycopg2.connect(
        host=DB_HOST, dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD, port=DB_PORT
    )
    load_users(conn)
    load_purchases(conn)
    conn.close()
    print("Import complete.")