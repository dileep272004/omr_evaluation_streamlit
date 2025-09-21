import sqlite3
conn = sqlite3.connect('omr.db')
c = conn.cursor()
c.execute("SELECT * FROM results WHERE student_id IN ('TestA', 'TestB')")
print(c.fetchall())
conn.close()