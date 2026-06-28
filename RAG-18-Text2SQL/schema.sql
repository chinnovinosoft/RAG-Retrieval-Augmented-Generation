-- ============================================================
--  E-Commerce Data Model  –  20 Tables
--  Use this file to set up your NeonDB schema.
--  Run: psql $DATABASE_URL -f schema.sql
-- ============================================================

-- ----------------------------------------------------------------
-- 1. DEPARTMENTS  –  company org structure
-- ----------------------------------------------------------------
CREATE TABLE IF NOT EXISTS departments (
    id          SERIAL PRIMARY KEY,
    name        VARCHAR(100) NOT NULL,
    budget      NUMERIC(15, 2),
    location    VARCHAR(100),
    created_at  TIMESTAMP DEFAULT NOW()
);

INSERT INTO departments (name, budget, location) VALUES
    ('Engineering',     500000.00,  'San Francisco'),
    ('Marketing',       200000.00,  'New York'),
    ('Operations',      350000.00,  'Austin'),
    ('Customer Support',150000.00,  'Chicago'),
    ('Finance',         180000.00,  'Boston');


-- ----------------------------------------------------------------
-- 2. EMPLOYEES  –  staff linked to departments
-- ----------------------------------------------------------------
CREATE TABLE IF NOT EXISTS employees (
    id             SERIAL PRIMARY KEY,
    first_name     VARCHAR(50)  NOT NULL,
    last_name      VARCHAR(50)  NOT NULL,
    email          VARCHAR(100) UNIQUE NOT NULL,
    department_id  INT REFERENCES departments(id),
    salary         NUMERIC(10, 2),
    hire_date      DATE,
    is_active      BOOLEAN DEFAULT TRUE
);

INSERT INTO employees (first_name, last_name, email, department_id, salary, hire_date) VALUES
    ('Alice',   'Johnson', 'alice.j@shop.com',   1, 120000.00, '2020-03-15'),
    ('Bob',     'Smith',   'bob.s@shop.com',     2,  90000.00, '2019-07-01'),
    ('Carol',   'Lee',     'carol.l@shop.com',   3,  85000.00, '2021-01-10'),
    ('David',   'Kim',     'david.k@shop.com',   4,  70000.00, '2022-05-20'),
    ('Eva',     'Patel',   'eva.p@shop.com',     1, 115000.00, '2020-11-30'),
    ('Frank',   'Brown',   'frank.b@shop.com',   5,  95000.00, '2018-09-14');


-- ----------------------------------------------------------------
-- 3. WAREHOUSES  –  physical storage locations
-- ----------------------------------------------------------------
CREATE TABLE IF NOT EXISTS warehouses (
    id          SERIAL PRIMARY KEY,
    name        VARCHAR(100) NOT NULL,
    city        VARCHAR(100),
    country     VARCHAR(50) DEFAULT 'USA',
    capacity    INT,           -- total storage units
    is_active   BOOLEAN DEFAULT TRUE
);

INSERT INTO warehouses (name, city, capacity) VALUES
    ('East Coast Hub',   'Newark',      50000),
    ('West Coast Hub',   'Los Angeles', 60000),
    ('Central Depot',    'Chicago',     45000),
    ('Southern Depot',   'Dallas',      40000),
    ('Pacific Northwest','Seattle',     30000);


-- ----------------------------------------------------------------
-- 4. SUPPLIERS  –  vendors who supply products
-- ----------------------------------------------------------------
CREATE TABLE IF NOT EXISTS suppliers (
    id            SERIAL PRIMARY KEY,
    name          VARCHAR(150) NOT NULL,
    contact_email VARCHAR(100),
    phone         VARCHAR(20),
    country       VARCHAR(50),
    rating        NUMERIC(3, 1),   -- 1.0 – 5.0
    is_active     BOOLEAN DEFAULT TRUE
);

INSERT INTO suppliers (name, contact_email, phone, country, rating) VALUES
    ('TechGadgets Inc.',   'supply@techgadgets.com', '800-111-2222', 'USA',   4.5),
    ('FashionForward Co.', 'orders@ffco.com',        '800-333-4444', 'Italy', 4.2),
    ('HomeEssentials Ltd.','b2b@homeessentials.com', '800-555-6666', 'China', 3.9),
    ('SportZone Corp.',    'wholesale@sportzone.com','800-777-8888', 'USA',   4.7),
    ('NaturalBeauty GmbH', 'export@naturalbeauty.de','800-999-0000', 'Germany',4.3);


-- ----------------------------------------------------------------
-- 5. CATEGORIES  –  product taxonomy (self-referencing for tree)
-- ----------------------------------------------------------------
CREATE TABLE IF NOT EXISTS categories (
    id          SERIAL PRIMARY KEY,
    name        VARCHAR(100) NOT NULL,
    parent_id   INT REFERENCES categories(id),   -- NULL means root category
    description TEXT
);

INSERT INTO categories (name, parent_id, description) VALUES
    ('Electronics',  NULL, 'All electronic devices and accessories'),
    ('Clothing',     NULL, 'Apparel for men, women, and kids'),
    ('Home & Garden',NULL, 'Products for home and outdoor living'),
    ('Sports',       NULL, 'Sports equipment and activewear'),
    ('Beauty',       NULL, 'Skincare, makeup and personal care'),
    ('Laptops',      1,    'Portable computing devices'),
    ('Phones',       1,    'Smartphones and accessories'),
    ('Mens Clothing',2,    'Shirts, pants, suits for men'),
    ('Womens Clothing',2,  'Dresses, tops, skirts for women'),
    ('Garden Tools', 3,    'Shovels, rakes, and outdoor tools');


-- ----------------------------------------------------------------
-- 6. PRODUCTS  –  core product catalog
-- ----------------------------------------------------------------
CREATE TABLE IF NOT EXISTS products (
    id           SERIAL PRIMARY KEY,
    name         VARCHAR(200) NOT NULL,
    description  TEXT,
    category_id  INT REFERENCES categories(id),
    supplier_id  INT REFERENCES suppliers(id),
    base_price   NUMERIC(10, 2),
    sku          VARCHAR(50) UNIQUE,
    is_active    BOOLEAN DEFAULT TRUE,
    created_at   TIMESTAMP DEFAULT NOW()
);

INSERT INTO products (name, description, category_id, supplier_id, base_price, sku) VALUES
    ('ProBook Laptop 15"',       'High-performance laptop for professionals', 6, 1, 1299.99, 'LPTP-001'),
    ('Galaxy Smartphone X',      '6.7" AMOLED, 256GB storage',               7, 1,  899.99, 'PHN-001'),
    ('Classic Oxford Shirt',     '100% cotton formal shirt',                  8, 2,   49.99, 'MEN-001'),
    ('Floral Summer Dress',      'Light and breezy summer dress',             9, 2,   65.99, 'WMN-001'),
    ('Stainless Garden Trowel',  'Rust-resistant garden trowel',             10, 3,   18.99, 'GDN-001'),
    ('Running Shoes Pro',        'Cushioned sole for long-distance running',  4, 4,  129.99, 'SPT-001'),
    ('Vitamin C Serum',          '20% pure vitamin C, anti-aging formula',   5, 5,   34.99, 'BTY-001'),
    ('Wireless Earbuds X1',      'Active noise cancellation, 30hr battery',  1, 1,  199.99, 'EARBUD-001'),
    ('Yoga Mat Premium',         '6mm non-slip TPE yoga mat',                4, 4,   45.99, 'SPT-002'),
    ('Smart Home Hub',           'Controls all your smart devices via voice', 1, 1,  149.99, 'SMRT-001');


-- ----------------------------------------------------------------
-- 7. PRODUCT_VARIANTS  –  size/color options per product
-- ----------------------------------------------------------------
CREATE TABLE IF NOT EXISTS product_variants (
    id           SERIAL PRIMARY KEY,
    product_id   INT REFERENCES products(id),
    size         VARCHAR(20),
    color        VARCHAR(50),
    extra_price  NUMERIC(8, 2) DEFAULT 0.00,   -- added on top of base_price
    variant_sku  VARCHAR(60) UNIQUE
);

INSERT INTO product_variants (product_id, size, color, extra_price, variant_sku) VALUES
    (3, 'S',  'White', 0.00,  'MEN-001-S-WHT'),
    (3, 'M',  'White', 0.00,  'MEN-001-M-WHT'),
    (3, 'L',  'Blue',  0.00,  'MEN-001-L-BLU'),
    (4, 'XS', 'Red',   0.00,  'WMN-001-XS-RED'),
    (4, 'S',  'Green', 0.00,  'WMN-001-S-GRN'),
    (6, '8',  'Black', 0.00,  'SPT-001-8-BLK'),
    (6, '9',  'Black', 0.00,  'SPT-001-9-BLK'),
    (6, '10', 'White', 5.00,  'SPT-001-10-WHT'),
    (9, NULL, 'Purple',0.00,  'SPT-002-PUR'),
    (9, NULL, 'Grey',  0.00,  'SPT-002-GRY');


-- ----------------------------------------------------------------
-- 8. INVENTORY  –  stock levels per product per warehouse
-- ----------------------------------------------------------------
CREATE TABLE IF NOT EXISTS inventory (
    id              SERIAL PRIMARY KEY,
    product_id      INT REFERENCES products(id),
    warehouse_id    INT REFERENCES warehouses(id),
    quantity        INT DEFAULT 0,
    reorder_level   INT DEFAULT 10,
    last_restocked  TIMESTAMP,
    UNIQUE (product_id, warehouse_id)
);

INSERT INTO inventory (product_id, warehouse_id, quantity, reorder_level, last_restocked) VALUES
    (1, 1, 200, 20, '2024-11-01'),
    (2, 1, 350, 30, '2024-11-05'),
    (3, 2, 500, 50, '2024-10-20'),
    (4, 2, 420, 40, '2024-10-22'),
    (5, 3, 800, 100,'2024-09-30'),
    (6, 3, 300, 30, '2024-11-10'),
    (7, 4, 600, 60, '2024-10-15'),
    (8, 1, 180, 20, '2024-11-12'),
    (9, 5, 250, 25, '2024-10-28'),
    (10,1, 120, 15, '2024-11-08');


-- ----------------------------------------------------------------
-- 9. CUSTOMERS  –  registered shoppers
-- ----------------------------------------------------------------
CREATE TABLE IF NOT EXISTS customers (
    id            SERIAL PRIMARY KEY,
    first_name    VARCHAR(50)  NOT NULL,
    last_name     VARCHAR(50)  NOT NULL,
    email         VARCHAR(100) UNIQUE NOT NULL,
    phone         VARCHAR(20),
    date_of_birth DATE,
    loyalty_points INT DEFAULT 0,
    created_at    TIMESTAMP DEFAULT NOW()
);

INSERT INTO customers (first_name, last_name, email, phone, date_of_birth, loyalty_points) VALUES
    ('James',   'Wilson',  'james.w@email.com',  '555-1001', '1990-04-12', 1200),
    ('Sophia',  'Garcia',  'sophia.g@email.com', '555-1002', '1985-08-23',  800),
    ('Liam',    'Martinez','liam.m@email.com',   '555-1003', '1992-01-30', 2500),
    ('Olivia',  'Chen',    'olivia.c@email.com', '555-1004', '1988-11-05',  350),
    ('Noah',    'Taylor',  'noah.t@email.com',   '555-1005', '1995-06-18', 4100),
    ('Emma',    'Anderson','emma.a@email.com',   '555-1006', '1991-03-27',  990);


-- ----------------------------------------------------------------
-- 10. ADDRESSES  –  customer shipping / billing addresses
-- ----------------------------------------------------------------
CREATE TABLE IF NOT EXISTS addresses (
    id            SERIAL PRIMARY KEY,
    customer_id   INT REFERENCES customers(id),
    address_type  VARCHAR(20) DEFAULT 'shipping',   -- 'shipping' or 'billing'
    street        VARCHAR(200),
    city          VARCHAR(100),
    state         VARCHAR(50),
    zip_code      VARCHAR(20),
    country       VARCHAR(50) DEFAULT 'USA',
    is_default    BOOLEAN DEFAULT FALSE
);

INSERT INTO addresses (customer_id, address_type, street, city, state, zip_code, is_default) VALUES
    (1, 'shipping', '123 Maple St',    'Boston',       'MA', '02101', TRUE),
    (2, 'shipping', '456 Oak Ave',     'Austin',       'TX', '78701', TRUE),
    (3, 'billing',  '789 Pine Rd',     'Seattle',      'WA', '98101', TRUE),
    (4, 'shipping', '321 Elm Blvd',    'Chicago',      'IL', '60601', TRUE),
    (5, 'shipping', '654 Cedar Lane',  'Los Angeles',  'CA', '90001', TRUE),
    (6, 'billing',  '987 Birch Court', 'New York',     'NY', '10001', TRUE);


-- ----------------------------------------------------------------
-- 11. ORDERS  –  order header / summary
-- ----------------------------------------------------------------
CREATE TABLE IF NOT EXISTS orders (
    id              SERIAL PRIMARY KEY,
    customer_id     INT REFERENCES customers(id),
    status          VARCHAR(30) DEFAULT 'pending',
    -- status: pending | confirmed | processing | shipped | delivered | cancelled
    total_amount    NUMERIC(12, 2),
    discount_amount NUMERIC(12, 2) DEFAULT 0.00,
    shipping_fee    NUMERIC(8, 2)  DEFAULT 0.00,
    address_id      INT REFERENCES addresses(id),
    ordered_at      TIMESTAMP DEFAULT NOW(),
    updated_at      TIMESTAMP DEFAULT NOW()
);

INSERT INTO orders (customer_id, status, total_amount, discount_amount, shipping_fee, address_id) VALUES
    (1, 'delivered',  1349.98, 50.00,  9.99, 1),
    (2, 'shipped',     899.99, 30.00,  0.00, 2),
    (3, 'processing',  179.97,  0.00,  5.99, 3),
    (4, 'pending',      65.99,  5.00,  4.99, 4),
    (5, 'delivered',   229.98,  0.00,  0.00, 5),
    (6, 'cancelled',   149.99, 10.00,  0.00, 6);


-- ----------------------------------------------------------------
-- 12. ORDER_ITEMS  –  line items within each order
-- ----------------------------------------------------------------
CREATE TABLE IF NOT EXISTS order_items (
    id          SERIAL PRIMARY KEY,
    order_id    INT REFERENCES orders(id),
    product_id  INT REFERENCES products(id),
    variant_id  INT REFERENCES product_variants(id),
    quantity    INT NOT NULL,
    unit_price  NUMERIC(10, 2) NOT NULL,
    subtotal    NUMERIC(12, 2) GENERATED ALWAYS AS (quantity * unit_price) STORED
);

INSERT INTO order_items (order_id, product_id, variant_id, quantity, unit_price) VALUES
    (1, 1, NULL, 1, 1299.99),
    (1, 5, NULL, 1,   18.99),
    (2, 2, NULL, 1,  899.99),
    (3, 6,  6,   1,  129.99),
    (3, 9,  9,   1,   45.99),
    (4, 4,  4,   1,   65.99),
    (5, 8, NULL, 1,  199.99),
    (5, 7, NULL, 1,   34.99),
    (6, 10,NULL, 1,  149.99);


-- ----------------------------------------------------------------
-- 13. PAYMENTS  –  payment transactions per order
-- ----------------------------------------------------------------
CREATE TABLE IF NOT EXISTS payments (
    id              SERIAL PRIMARY KEY,
    order_id        INT REFERENCES orders(id),
    amount          NUMERIC(12, 2) NOT NULL,
    method          VARCHAR(30),    -- 'credit_card' | 'paypal' | 'stripe' | 'bank_transfer'
    status          VARCHAR(20) DEFAULT 'pending',  -- 'pending' | 'completed' | 'failed' | 'refunded'
    transaction_ref VARCHAR(100),
    paid_at         TIMESTAMP
);

INSERT INTO payments (order_id, amount, method, status, transaction_ref, paid_at) VALUES
    (1, 1309.97, 'credit_card',   'completed', 'TXN-A001', '2024-10-01 10:15:00'),
    (2,  869.99, 'paypal',        'completed', 'TXN-A002', '2024-10-15 14:22:00'),
    (3,  185.96, 'stripe',        'completed', 'TXN-A003', '2024-11-01 09:05:00'),
    (4,   64.98, 'credit_card',   'pending',   'TXN-A004', NULL),
    (5,  229.98, 'bank_transfer', 'completed', 'TXN-A005', '2024-09-20 16:45:00'),
    (6,  139.99, 'paypal',        'refunded',  'TXN-A006', '2024-08-10 11:30:00');


-- ----------------------------------------------------------------
-- 14. SHIPPING  –  shipment tracking per order
-- ----------------------------------------------------------------
CREATE TABLE IF NOT EXISTS shipping (
    id              SERIAL PRIMARY KEY,
    order_id        INT REFERENCES orders(id) UNIQUE,
    carrier         VARCHAR(50),    -- 'FedEx' | 'UPS' | 'USPS' | 'DHL'
    tracking_number VARCHAR(100),
    status          VARCHAR(30) DEFAULT 'not_shipped',
    -- not_shipped | picked_up | in_transit | out_for_delivery | delivered
    shipped_at      TIMESTAMP,
    estimated_delivery DATE,
    delivered_at    TIMESTAMP
);

INSERT INTO shipping (order_id, carrier, tracking_number, status, shipped_at, estimated_delivery, delivered_at) VALUES
    (1, 'FedEx', 'FX1234567890', 'delivered',    '2024-10-02', '2024-10-05', '2024-10-04 14:30:00'),
    (2, 'UPS',   'UPS987654321', 'in_transit',   '2024-10-16', '2024-10-20', NULL),
    (3, 'USPS',  'USPS112233445','picked_up',    '2024-11-02', '2024-11-07', NULL),
    (5, 'DHL',   'DHL556677889', 'delivered',    '2024-09-21', '2024-09-25', '2024-09-24 10:00:00'),
    (6, 'FedEx', 'FX0000000001', 'not_shipped',  NULL,          NULL,         NULL);


-- ----------------------------------------------------------------
-- 15. RETURNS  –  return request header
-- ----------------------------------------------------------------
CREATE TABLE IF NOT EXISTS returns (
    id              SERIAL PRIMARY KEY,
    order_id        INT REFERENCES orders(id),
    customer_id     INT REFERENCES customers(id),
    reason          TEXT,
    status          VARCHAR(30) DEFAULT 'requested',
    -- requested | approved | rejected | received | refunded
    requested_at    TIMESTAMP DEFAULT NOW(),
    resolved_at     TIMESTAMP
);

INSERT INTO returns (order_id, customer_id, reason, status, resolved_at) VALUES
    (6, 6, 'Item arrived damaged',         'refunded',  '2024-08-15 09:00:00'),
    (1, 1, 'Wrong item shipped',           'received',  NULL),
    (3, 3, 'Changed my mind',              'approved',  NULL),
    (2, 2, 'Product not as described',     'requested', NULL),
    (5, 5, 'Defective product',            'rejected',  '2024-10-01 12:00:00');


-- ----------------------------------------------------------------
-- 16. RETURN_ITEMS  –  individual items in a return
-- ----------------------------------------------------------------
CREATE TABLE IF NOT EXISTS return_items (
    id            SERIAL PRIMARY KEY,
    return_id     INT REFERENCES returns(id),
    order_item_id INT REFERENCES order_items(id),
    quantity      INT NOT NULL,
    refund_amount NUMERIC(10, 2)
);

INSERT INTO return_items (return_id, order_item_id, quantity, refund_amount) VALUES
    (1, 9,  1, 149.99),
    (2, 1,  1, 1299.99),
    (3, 4,  1, 129.99),
    (4, 3,  1, 899.99),
    (5, 7,  1, 199.99);


-- ----------------------------------------------------------------
-- 17. REVIEWS  –  customer product reviews
-- ----------------------------------------------------------------
CREATE TABLE IF NOT EXISTS reviews (
    id          SERIAL PRIMARY KEY,
    product_id  INT REFERENCES products(id),
    customer_id INT REFERENCES customers(id),
    rating      INT CHECK (rating BETWEEN 1 AND 5),
    title       VARCHAR(150),
    body        TEXT,
    is_verified BOOLEAN DEFAULT FALSE,   -- verified purchase
    created_at  TIMESTAMP DEFAULT NOW()
);

INSERT INTO reviews (product_id, customer_id, rating, title, body, is_verified) VALUES
    (1, 1, 5, 'Best laptop I have ever owned',    'Super fast, great battery life.',              TRUE),
    (2, 2, 4, 'Great phone, minor issues',        'Camera is amazing but gets warm under load.', TRUE),
    (6, 3, 5, 'Perfect running shoes',            'Very comfortable for marathons.',              TRUE),
    (7, 4, 3, 'Decent serum',                    'Saw some improvement after 4 weeks.',          TRUE),
    (8, 5, 5, 'Outstanding earbuds',             'Best ANC I have experienced.',                 TRUE),
    (9, 6, 4, 'Great yoga mat',                  'Non-slip and thick enough for knees.',         TRUE);


-- ----------------------------------------------------------------
-- 18. WISHLISTS  –  items customers saved for later
-- ----------------------------------------------------------------
CREATE TABLE IF NOT EXISTS wishlists (
    id          SERIAL PRIMARY KEY,
    customer_id INT REFERENCES customers(id),
    product_id  INT REFERENCES products(id),
    added_at    TIMESTAMP DEFAULT NOW(),
    UNIQUE (customer_id, product_id)
);

INSERT INTO wishlists (customer_id, product_id) VALUES
    (1, 8),
    (1, 10),
    (2, 1),
    (3, 2),
    (4, 6),
    (5, 7),
    (6, 3);


-- ----------------------------------------------------------------
-- 19. COUPONS  –  discount codes
-- ----------------------------------------------------------------
CREATE TABLE IF NOT EXISTS coupons (
    id              SERIAL PRIMARY KEY,
    code            VARCHAR(30) UNIQUE NOT NULL,
    discount_type   VARCHAR(20),    -- 'percentage' | 'fixed'
    discount_value  NUMERIC(8, 2),  -- 20 means 20% or $20 off
    min_order_value NUMERIC(10, 2) DEFAULT 0.00,
    max_uses        INT,
    used_count      INT DEFAULT 0,
    valid_from      DATE,
    valid_until     DATE,
    is_active       BOOLEAN DEFAULT TRUE
);

INSERT INTO coupons (code, discount_type, discount_value, min_order_value, max_uses, valid_from, valid_until) VALUES
    ('WELCOME10',  'percentage', 10.00,  50.00, 1000, '2024-01-01', '2024-12-31'),
    ('SUMMER20',   'percentage', 20.00, 100.00,  500, '2024-06-01', '2024-08-31'),
    ('FLAT50',     'fixed',      50.00, 200.00,  200, '2024-10-01', '2024-10-31'),
    ('LOYAL25',    'percentage', 25.00, 150.00,  300, '2024-11-01', '2024-11-30'),
    ('NEWUSER5',   'fixed',       5.00,   0.00, 5000, '2024-01-01', '2025-12-31');


-- ----------------------------------------------------------------
-- 20. AUDIT_LOGS  –  system event trail
-- ----------------------------------------------------------------
CREATE TABLE IF NOT EXISTS audit_logs (
    id          SERIAL PRIMARY KEY,
    table_name  VARCHAR(50),
    action      VARCHAR(10),    -- INSERT | UPDATE | DELETE
    record_id   INT,
    changed_by  VARCHAR(100),   -- email or system process name
    old_values  JSONB,
    new_values  JSONB,
    happened_at TIMESTAMP DEFAULT NOW()
);

INSERT INTO audit_logs (table_name, action, record_id, changed_by, old_values, new_values) VALUES
    ('orders',   'UPDATE', 1, 'system',        '{"status":"processing"}', '{"status":"delivered"}'),
    ('inventory','UPDATE', 1, 'system',        '{"quantity":250}',         '{"quantity":200}'),
    ('coupons',  'UPDATE', 1, 'admin@shop.com','{"used_count":0}',         '{"used_count":1}'),
    ('products', 'UPDATE', 3, 'alice.j@shop.com','{"base_price":54.99}',   '{"base_price":49.99}'),
    ('returns',  'INSERT', 1, 'system',        NULL,                       '{"status":"requested"}');
