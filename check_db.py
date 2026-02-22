import sqlite3
import os

DB_NAME = "database.db"

if not os.path.exists(DB_NAME):
    print(f"Error: {DB_NAME} not found.")
else:
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("PRAGMA table_info(users)")
    columns = c.fetchall()
    print("Columns in 'users' table:")
    for col in columns:
        print(col)
    conn.close()
