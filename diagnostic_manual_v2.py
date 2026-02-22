import sqlite3
import os
import sys
from nutrition_service import get_nutrition

DB_NAME = "database.db"
OUTPUT_FILE = "debug_output.txt"

def log(msg):
    print(msg)
    with open(OUTPUT_FILE, "a") as f:
        f.write(str(msg) + "\n")

if os.path.exists(OUTPUT_FILE):
    os.remove(OUTPUT_FILE)

log("Starting diagnosis...")

def check_db():
    log(f"Checking database: {DB_NAME}")
    if not os.path.exists(DB_NAME):
        log("Database file NOT FOUND!")
        return
    
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    
    try:
        c.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = c.fetchall()
        log(f"Tables found: {[t['name'] for t in tables]}")
        
        if 'daily_logs' in [t['name'] for t in tables]:
            c.execute("PRAGMA table_info(daily_logs);")
            columns = c.fetchall()
            log(f"Columns in daily_logs: {[col['name'] for col in columns]}")
            
            c.execute("SELECT * FROM daily_logs;")
            logs = c.fetchall()
            log(f"Total rows in daily_logs: {len(logs)}")
            for log_row in logs[-10:]:
                log(dict(log_row))
        else:
            log("daily_logs table NOT FOUND!")
            
    except Exception as e:
        log(f"Error in check_db: {e}")
    finally:
        conn.close()

def check_lookup():
    log("\nChecking lookup for 'apple':")
    try:
        res = get_nutrition("apple")
        log(res)
    except Exception as e:
        log(f"Lookup error (apple): {e}")
    
    log("\nChecking lookup for 'banana' (Standard):")
    try:
        res = get_nutrition("banana", portion_size=1.0)
        log(res)
    except Exception as e:
        log(f"Lookup error (banana): {e}")

if __name__ == "__main__":
    check_db()
    check_lookup()
    log("\nDiagnosis complete.")
