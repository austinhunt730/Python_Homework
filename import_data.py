#!/usr/bin/env python3
import os, sys, glob, csv, uuid
import psycopg2

# ---- config: edit or use env vars ----
DB_NAME = os.getenv("PGDATABASE", "ecommerce")
DB_USER = os.getenv("PGUSER", "ecommerce_user")
DB_PASS = os.getenv("PGPASSWORD", "temp_pw")
DB_HOST = os.getenv("PGHOST", "localhost")
DB_PORT = int(os.getenv("PGPORT", "5432"))

# Glob patterns for your files
PURCHASES_GLOBS = ["data/purchase_data/*.csv"]
USERS_GLOBS     = ["data/user_data/*.csv"]

# Conflict targets
PURCHASES_CONFLICT_COLS = ["transaction_id"]
USERS_CONFLICT_COLS     = ["email"]

def find_files(patterns):
    files = []
    for p in patterns:
        files.extend(glob.glob(p, recursive=True))
    # de-dup and keep stable order
    seen, out = set(), []
    for f in files:
        fp = os.path.abspath(f)
        if fp not in seen and fp.lower().endswith(".csv"):
            seen.add(fp); out.append(fp)
    return out

def read_header_cols(path):
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
    # normalize to lowercase and strip spaces to match SQL identifiers
    return [h.strip() for h in header]

def copy_into_temp(cur, table, csv_path, cols):
    temp = f"tmp_{table}_{uuid.uuid4().hex[:8]}"
    # create temp as structure clone
    cur.execute(f'CREATE TEMP TABLE "{temp}" AS SELECT * FROM {table} WITH NO DATA;')
    # COPY only the CSV columns (exclude identity like users.user_id)
    col_list = ", ".join([f'"{c}"' for c in cols])
    copy_sql = f'COPY "{temp}" ({col_list}) FROM STDIN WITH (FORMAT csv, HEADER true)'
    with open(csv_path, "r", encoding="utf-8") as f:
        cur.copy_expert(copy_sql, f)
    return temp

def insert_from_temp(cur, table, temp, cols, conflict_cols):
    col_list = ", ".join([f'"{c}"' for c in cols])
    placeholders = ", ".join([f'"{c}"' for c in cols])
    conflict = ", ".join([f'"{c}"' for c in conflict_cols]) if conflict_cols else None
    if conflict:
        sql = f'INSERT INTO {table} ({col_list}) SELECT {placeholders} FROM "{temp}" ON CONFLICT ({conflict}) DO NOTHING;'
    else:
        sql = f'INSERT INTO {table} ({col_list}) SELECT {placeholders} FROM "{temp}";'
    cur.execute(sql)

def import_csv_group(conn, table, files, conflict_cols):
    if not files:
        print(f"[skip] No files for {table}")
        return
    with conn.cursor() as cur:
        for path in files:
            print(f"[load] {table} <- {path}")
            cols = read_header_cols(path)
            # Verify columns exist in target
            cur.execute("""
                SELECT column_name
                FROM information_schema.columns
                WHERE table_schema = 'public' AND table_name = %s
            """, (table,))
            valid = {r[0] for r in cur.fetchall()}
            bad = [c for c in cols if c not in valid]
            if bad:
                raise ValueError(f"{path}: unknown columns for {table}: {bad}")
            conn.rollback(); conn.autocommit = False
            try:
                temp = copy_into_temp(cur, table, path, cols)
                insert_from_temp(cur, table, temp, cols, conflict_cols)
                cur.execute(f'DROP TABLE IF EXISTS "{temp}";')
                conn.commit()
            except Exception as e:
                conn.rollback()
                print(f"[error] {path}: {e}")
                raise

def main():
    conn = psycopg2.connect(
        dbname=DB_NAME, user=DB_USER, password=DB_PASS, host=DB_HOST, port=DB_PORT
    )
    try:
        purchases_files = find_files(PURCHASES_GLOBS)
        users_files     = find_files(USERS_GLOBS)
        import_csv_group(conn, "purchases", purchases_files, PURCHASES_CONFLICT_COLS)
        import_csv_group(conn, "users", users_files, USERS_CONFLICT_COLS)
    finally:
        conn.close()

if __name__ == "__main__":
    if "--help" in sys.argv or "-h" in sys.argv:
        print("Usage: set PG* env vars or edit script, then run: python import_csvs.py")
        sys.exit(0)
    main()
