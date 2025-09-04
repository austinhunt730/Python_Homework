CREATE TABLE IF NOT EXISTS users(
  user_id      TEXT PRIMARY KEY,
  signup_date  DATE,
  region       TEXT
);

CREATE TABLE IF NOT EXISTS purchases(
  purchase_id  TEXT PRIMARY KEY,
  user_id      TEXT REFERENCES users(user_id),
  product_id   TEXT,
  category     TEXT,
  price        NUMERIC,
  quantity     INT,
  purchased_at TIMESTAMP
);