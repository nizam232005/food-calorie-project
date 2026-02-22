import sqlite3

DB_NAME = "database.db"

def migrate():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    
    # List of columns to add and their types
    # (name, type)
    new_columns = [
        ("age", "INTEGER"),
        ("height", "INTEGER"),
        ("weight", "INTEGER"),
        ("gender", "TEXT"),
        ("activity_level", "TEXT"),
        ("goal", "TEXT"),
        ("diet", "TEXT")
    ]
    
    # Get existing columns
    c.execute("PRAGMA table_info(users)")
    existing_columns = [row[1] for row in c.fetchall()]
    
    print(f"Existing columns: {existing_columns}")
    
    for col_name, col_type in new_columns:
        if col_name not in existing_columns:
            print(f"Adding column: {col_name}")
            try:
                c.execute(f"ALTER TABLE users ADD COLUMN {col_name} {col_type}")
                print(f"Successfully added {col_name}")
            except sqlite3.OperationalError as e:
                print(f"Error adding {col_name}: {e}")
        else:
            print(f"Column {col_name} already exists.")
            
    conn.commit()
    conn.close()
    print("Migration complete.")

if __name__ == "__main__":
    migrate()
