#!/usr/bin/env python3
"""
Robust CSV → Postgres loader for the e-commerce project.

It:
  1) Loads users from data/user_data/*.csv and picks one identifier column
     for users.user_id with priority: user_id → email → id → row-index.
  2) Auto-detects which purchases column links to users by measuring overlap
     with the set of loaded user_ids (works for buyer_id, customer_email, uid, etc.).
  3) Normalizes to:
       users(user_id, signup_date, region)
       purchases(purchase_id, user_id, product_id, category, price, quantity, purchased_at)
  4) Filters purchases to only rows whose user_id exists in users (FK-safe).
"""

import os, glob
from io import StringIO
from typing import Optional, Tuple

import pandas as pd
import psycopg2

# ---- DB config via env vars ----
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_NAME = os.getenv("DB_NAME", "ecommerce")
DB_USER = os.getenv("DB_USER", "ecommerce_user")
DB_PASSWORD = os.getenv("DB_PASSWORD", "temp_pw")
DB_PORT = int(os.getenv("DB_PORT", "5432"))

# ---- util: read & copy ----
def read_concat(pattern: str) -> pd.DataFrame:
    files = sorted(glob.glob(pattern))
    if not files:
        return pd.DataFrame()
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

def copy_df(conn, df: pd.DataFrame, table: str, columns: list[str]):
    # Reorder/limit columns
    df = df.loc[:, columns]
    # Stream via CSV to COPY
    buf = StringIO()
    df.to_csv(buf, index=False, header=True)
    buf.seek(0)
    with conn.cursor() as cur:
        col_list = f"({', '.join(columns)})"
        cur.copy_expert(f"COPY {table} {col_list} FROM STDIN WITH CSV HEADER", buf)
    conn.commit()

# ---- USERS ----
def choose_users_id_column(df: pd.DataFrame) -> str:
    for c in ["user_id", "email", "id"]:
        if c in df.columns:
            return c
    return "__row__"  # fallback to row-index

def make_users(users_raw: pd.DataFrame) -> pd.DataFrame:
    if users_raw.empty:
        raise RuntimeError("No user CSVs found under data/user_data/*.csv")

    id_col = choose_users_id_column(users_raw)
    if id_col == "__row__":
        uid = users_raw.index.astype(str)
    else:
        uid = users_raw[id_col].fillna("").astype(str)

    # signup_date candidates
    sd = None
    for cand in [
        "signup_date","created_at","signup_datetime","signup_time",
        "account_created","signup_timestamp","date_joined"
    ]:
        if cand in users_raw.columns:
            sd = pd.to_datetime(users_raw[cand], errors="coerce").dt.date
            break
    if sd is None:
        sd = pd.NaT

    # region candidates
    if   "region"   in users_raw.columns: region = users_raw["region"].astype(str)
    elif "state"    in users_raw.columns: region = users_raw["state"].astype(str)
    elif "country"  in users_raw.columns: region = users_raw["country"].astype(str)
    elif "location" in users_raw.columns: region = users_raw["location"].astype(str)
    else:                                  region = "unknown"

    users_norm = pd.DataFrame({
        "user_id": uid,
        "signup_date": sd,
        "region": region
    }).drop_duplicates(subset=["user_id"])

    print(f"[info] users identifier column: {id_col!r}  → unique users: {len(users_norm):,}")
    return users_norm

# ---- PURCHASES ----
def pick(df: pd.DataFrame, names: list[str]) -> Optional[str]:
    for n in names:
        if n in df.columns: return n
    return None

def detect_purchase_user_column(purchases_raw: pd.DataFrame, user_ids: set[str]) -> Optional[str]:
    """
    Find which purchases column best matches users.user_id by overlap.
    We check common names first, then all object-like columns.
    """
    if purchases_raw.empty: return None

    seed = [
        "user_id","buyer_id","customer_id","uid",
        "email","user_email","customer_email","buyer_email",
        "UserID","userId","USER_ID","user","customer","buyer","id"
    ]
    candidates: list[str] = []
    seen = set()

    for c in seed:
        if c in purchases_raw.columns and c not in seen:
            candidates.append(c); seen.add(c)

    # add all object dtype columns (strings)
    for c in purchases_raw.columns:
        if c not in seen and purchases_raw[c].dtype == "object":
            candidates.append(c); seen.add(c)

    best_col, best_overlap = None, 0
    s_users = user_ids  # set for O(1) lookup

    for c in candidates:
        try:
            s = purchases_raw[c].astype(str)
        except Exception:
            continue
        overlap = s.isin(s_users).sum()
        if overlap > best_overlap:
            best_col, best_overlap = c, overlap

    if best_col:
        print(f"[info] purchases→users link column: {best_col!r}  (matched rows: {best_overlap:,})")
    else:
        print("[warn] could not detect a purchases user link column by overlap")
    return best_col

def make_purchases(purchases_raw: pd.DataFrame, user_col: Optional[str]) -> pd.DataFrame:
    if purchases_raw.empty:
        return pd.DataFrame(columns=["purchase_id","user_id","product_id","category","price","quantity","purchased_at"])

    p_purchase_id  = pick(purchases_raw, ["purchase_id","order_id","id"])
    p_product_id   = pick(purchases_raw, ["product_id","sku","item_id"])
    p_category     = pick(purchases_raw, ["category","cat","dept"])
    p_price        = pick(purchases_raw, ["price","amount","revenue"])
    p_quantity     = pick(purchases_raw, ["quantity","qty","count"])
    p_purchased_at = pick(purchases_raw, ["purchased_at","timestamp","purchase_time","order_time","created_at"])

    shaped = pd.DataFrame({
        "purchase_id":  purchases_raw[p_purchase_id].astype(str) if p_purchase_id else purchases_raw.index.astype(str),
        "user_id":      purchases_raw[user_col].fillna("").astype(str) if user_col and user_col in purchases_raw.columns else "",
        "product_id":   purchases_raw[p_product_id].astype(str) if p_product_id else "",
        "category":     purchases_raw[p_category].astype(str) if p_category else "",
        "price":        pd.to_numeric(purchases_raw[p_price], errors="coerce") if p_price else 0.0,
        "quantity":     pd.to_numeric(purchases_raw[p_quantity], errors="coerce") if p_quantity else 1,
        "purchased_at": pd.to_datetime(purchases_raw[p_purchased_at], errors="coerce") if p_purchased_at else pd.NaT,
    })

    # basic cleanup
    shaped["quantity"] = shaped["quantity"].fillna(1).astype("Int64")
    shaped["price"] = shaped["price"].fillna(0.0)
    shaped = shaped.drop_duplicates(subset=["purchase_id"])
    return shaped

# ---- MAIN ----
def main():
    users_raw = read_concat("data/user_data/*.csv")
    purchases_raw = read_concat("data/purchase_data/*.csv")

    # Build & load users
    users_norm = make_users(users_raw)
    conn = psycopg2.connect(host=DB_HOST, dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD, port=DB_PORT)
    try:
        copy_df(conn, users_norm, "users", ["user_id","signup_date","region"])
        print(f"Loaded users: {len(users_norm):,}")
        valid_user_ids = set(users_norm["user_id"].astype(str))
        # Prepare purchases
        user_col = detect_purchase_user_column(purchases_raw, valid_user_ids)
        purchases_norm = make_purchases(purchases_raw, user_col)

        if purchases_norm.empty:
            print("Loaded purchases: 0 (no purchases data or no link column)")
            return

        # FK safety: only keep purchases whose user exists
        before = len(purchases_norm)
        purchases_norm = purchases_norm[purchases_norm["user_id"].isin(valid_user_ids)].copy()
        dropped = before - len(purchases_norm)
        if dropped > 0:
            print(f"Dropped {dropped:,} purchases without matching users (FK filter).")

        copy_df(conn, purchases_norm, "purchases",
                ["purchase_id","user_id","product_id","category","price","quantity","purchased_at"])
        print(f"Loaded purchases: {len(purchases_norm):,}")
    finally:
        conn.close()
        print("Import complete.")

if __name__ == "__main__":
    main()
