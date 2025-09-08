import psycopg2

# Connection details – adjust to your setup
conn = psycopg2.connect(
    dbname="ecommerce",
    user="ecommerce_user",
    password="temp_pw",
    host="localhost",
    port=5432
)

cur = conn.cursor()

schema_sql = """
DROP TABLE IF EXISTS purchases;
DROP TABLE IF EXISTS users;

CREATE TABLE purchases (
    transaction_id     VARCHAR(100) PRIMARY KEY,
    user_email         VARCHAR(255) NOT NULL,
    product_name       VARCHAR(255) NOT NULL,
    product_category   VARCHAR(100),
    quantity           INT NOT NULL,
    unit_price         DECIMAL(10,2) NOT NULL,
    discount_percent   DECIMAL(5,2),
    discount_amount    DECIMAL(10,2),
    shipping_cost      DECIMAL(10,2),
    total_price        DECIMAL(12,2) NOT NULL,
    purchase_date      DATE NOT NULL,
    purchase_time      TIME,
    payment_method     VARCHAR(50),
    purchase_status    VARCHAR(50),
    month              INT,
    year               INT
);

CREATE TABLE users (
    user_id            BIGINT PRIMARY KEY GENERATED ALWAYS AS IDENTITY,
    first_name         VARCHAR(100) NOT NULL,
    last_name          VARCHAR(100) NOT NULL,
    email              VARCHAR(255) UNIQUE NOT NULL,
    password_hash      VARCHAR(255) NOT NULL,
    phone_number       VARCHAR(20),
    date_of_birth      DATE,
    time_on_app        INTERVAL,
    user_type          VARCHAR(50),
    is_active          BOOLEAN DEFAULT TRUE,
    last_payment_method VARCHAR(50),
    reviews            VARCHAR(1000),
    last_ip            VARCHAR(45),
    last_coordinates   VARCHAR(100),
    last_device        VARCHAR(100),
    last_browser       VARCHAR(100),
    last_os            VARCHAR(100),
    last_login         TIMESTAMP,
    last_logout        TIMESTAMP,
    in_cart            VARCHAR(255),
    wishlist           VARCHAR(255),
    last_search        TEXT,
    created_date       TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    generated_at       TIMESTAMP,
    purchase_count     INT DEFAULT 0,
    total_spent        DECIMAL(12,2) DEFAULT 0.00
);
"""

cur.execute(schema_sql)
conn.commit()

cur.close()
conn.close()

print("Tables created successfully.")
