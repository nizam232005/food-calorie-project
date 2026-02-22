import sqlite3
import os

DB_PATH = r"c:\food_calorie_project\database.db"
LOG_PATH = r"c:\food_calorie_project\schema_report.txt"

def check():
    report = []
    report.append(f"DB Path: {DB_PATH}")
    if not os.path.exists(DB_PATH):
        report.append("DB DOES NOT EXIST")
    else:
        try:
            conn = sqlite3.connect(DB_PATH)
            c = conn.cursor()
            c.execute("PRAGMA table_info(users)")
            cols = c.fetchall()
            report.append("Columns in 'users':")
            for col in cols:
                report.append(str(col))
            conn.close()
        except Exception as e:
            report.append(f"ERROR: {e}")
            
    with open(LOG_PATH, "w") as f:
        f.write("\n".join(report))

if __name__ == "__main__":
    check()
