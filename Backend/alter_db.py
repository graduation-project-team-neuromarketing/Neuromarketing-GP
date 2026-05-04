import sqlite3
import os

db_path = "neuromarketing.db"
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

try:
    cursor.execute("ALTER TABLE companies ADD COLUMN category_id INTEGER REFERENCES categories(id);")
    print("Column category_id added successfully.")
except sqlite3.OperationalError as e:
    print(f"Error adding column: {e}")

conn.commit()
conn.close()

from database import engine
import models
models.Base.metadata.create_all(bind=engine)
print("Tables created/verified.")
