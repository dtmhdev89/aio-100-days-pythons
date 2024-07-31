import sqlite3
import pandas as pd

connection = sqlite3.connect("db.sqlite")
cursor = connection.cursor()

# cursor.execute("""
#     CREATE TABLE CUSTOMERS (
#         EMAIL TEXT PRIMARY KEY,
#         NAME TEXT NOT NULL,
#         PHONE TEXT NOT NULL
#     );
# """)

# cursor.execute("""
#     INSERT INTO CUSTOMERS(EMAIL, NAME, PHONE)
#     VALUES
#         ('email1@aivietname.edu.vn', 'name1', '012345'),
#         ('email2@aivietname.edu.vn', 'name2', '012342345')
#     ;
# """)

# cursor.execute("""
#     UPDATE CUSTOMERS
#         SET NAME = "name11"
#         WHERE 1 = 1
#         AND EMAIL = 'email1@aivietname.edu.vn'
#     ;
# """)

# cursor.execute("""
#     DELETE FROM CUSTOMERS
#     WHERE 1 = 1
#     AND EMAIL = 'email1@aivietname.edu.vn'
#     ;
# """)

# cursor.execute("""
#     CREATE TABLE PRODUCTS (
#         ID INTEGER PRIMARY KEY,
#         NAME TEXT NOT NULL,
#         PRICE INTEGER NOT NULL
#     )
#     ;
# """)

# cursor.execute("""
#     INSERT INTO PRODUCTS (ID, NAME, PRICE)
#     VALUES
#         (1, 'iPhone15', 18000000),
#         (2, 'Galaxy Z-Fold 5', 30000000)
#     ;
# """)

# cursor.execute("""
#     UPDATE PRODUCTS
#     SET PRICE = 50000000
#     WHERE ID = 2
#     ;
# """)

# cursor.execute("""
#     DELETE FROM PRODUCTS
#     WHERE ID = 1
#     ;
# """)

connection.commit()

data = pd.read_sql_query("SELECT * FROM CUSTOMERS", connection)
print(data)

data = pd.read_sql_query("SELECT * FROM PRODUCTS", connection)
print(data)
