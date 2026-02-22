import sqlite3
import os
from nutrition_service import get_nutrition

DB_NAME = "database.db"

def check_db():
    print(f"Checking database: {DB_NAME}")
    if not os.path.exists(DB_NAME):
        print("Database file NOT FOUND!")
        return
    
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    
    try:
        c.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = c.fetchall()
        print(f"Tables found: {[t['name'] for t in tables]}")
        
        if 'daily_logs' in [t['name'] for t in tables]:
            c.execute("SELECT * FROM daily_logs;")
            logs = c.fetchall()
            print(f"Total rows in daily_logs: {len(logs)}")
            for log in logs[-5:]:
                print(dict(log))
        else:
            print("daily_logs table NOT FOUND!")
            
    except Exception as e:
        print(f"Error: {e}")
    finally:
        conn.close()

def check_lookup():
    print("\nChecking lookup for 'apple':")
    print(get_nutrition("apple"))
    
    print("\nChecking lookup for 'pizza' (Standard):")
    print(get_nutrition("pizza", portion_size=1.0))

if __name__ == "__main__":
    check_db()
    check_lookup()
