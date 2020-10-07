import sqlite3

def db_execute_query(query):
    connection = sqlite3.connect('db.sqlite')
    cursor = connection.cursor()
    try:
        cursor.execute(query)
        connection.commit()
        last_id = cursor.lastrowid
        connection.close()
        return last_id
    except OperationalError as e:
        print(f"The error '{e}' occurred")

def db_read_query(query):
    connection = sqlite3.connect('db.sqlite')
    cursor = connection.cursor()
    result = None
    try:
        cursor.execute(query)
        result = cursor.fetchall()
        connection.close()
        return result
    except Error as e:
        print(f"The error '{e}' occurred")


"""
curl --header "Content-Type: application/json" \
  --request POST \
  --data '{"access":"0","start_time":1600874457,"endtime":1600874557,"place":"0","zone":"0","videostream":"0","objective":"0","active":1}' \
  http://0.0.0.0:5000/api

CREATE TABLE IF NOT EXISTS requests_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    access TEXT NOT NULL,
    start_time timestamp,
    endtime timestamp,
    place TEXT NOT NULL,
    zone TEXT NOT NULL,
    videostream TEXT NOT NULL,
    objective TEXT NOT NULL,
    active INT NOT NULL
);
"""
