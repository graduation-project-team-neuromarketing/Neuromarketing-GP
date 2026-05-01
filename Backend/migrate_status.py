import sqlite3
import os

def run_migration():
    db_path = 'neuromarketing.db'
    if os.path.exists(db_path):
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        try:
            cursor.execute("ALTER TABLE history ADD COLUMN status VARCHAR DEFAULT 'In Processing';")
            conn.commit()
            print("Migration successful: added 'status' column to history table.")
        except sqlite3.OperationalError as e:
            print(f"Migration error: {e}")
        finally:
            conn.close()
    else:
        print("Database not found")

if __name__ == "__main__":
    run_migration()
