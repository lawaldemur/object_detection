import sqlite3
import json
import os


with open(os.getcwd() + '/config.json') as json_config:
    config = json.load(json_config)['server']


def db_execute_query(query):
    connection = sqlite3.connect(config['db'])
    cursor = connection.cursor()
    
    cursor.execute(query)
    connection.commit()
    last_id = cursor.lastrowid
    connection.close()
    return last_id

def db_read_query(query):
    connection = sqlite3.connect(config['db'])
    cursor = connection.cursor()
    result = None

    cursor.execute(query)
    result = cursor.fetchall()
    connection.close()
    return result


def db_task_info(id):
    connection = sqlite3.connect(config['db'])
    cursor = connection.cursor()
    result = None

    cursor.execute("SELECT * FROM requests_log WHERE id = {} ORDER BY id DESC LIMIT 1".format(id))
    result = cursor.fetchone()
    connection.close()
    return result


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
