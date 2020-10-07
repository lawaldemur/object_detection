# gunicorn --threads 5 --workers 1 --bind 0.0.0.0:5000 wsgi:app
import request
import os, time, sched, json, requests
from importlib import import_module
from flask import Flask, render_template, Response, request
from threading import Thread
from pyngrok import ngrok
from camera import Camera
from db_connection import *
from detect_video import *


# class LoopThread(Thread):
#     def __init__(self):
#         Thread.__init__(self)

#     def run(self):
#         while True:
#             print('start detection (in loop)')
#             detection()


app = Flask(__name__)
# t_detection = LoopThread()
# t_detection.start()

# url = ngrok.connect(5000)
# print(' * Tunnel URL:', url)


@app.route('/')
def index():
    # Video streaming home page.
    return render_template('index.html')


def gen(camera, id):
    """Video streaming generator function."""
    while True:
        frame = camera.get_frame(id)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


@app.route('/stream/<id>')
def stream(id):
    """Video streaming route"""
    return Response(gen(Camera(), id),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/api', methods=['POST'])
def recieve_api_request():
    # receive data in json format
    req_data = request.get_json()

    # insert data to db
    query = """
        INSERT INTO
          requests_log (access, start_time, endtime, place, zone, videostream, objective, active)
        VALUES
          (\'{}\', {}, {}, \'{}\', \'{}\', \'{}\', \'{}\', {});
    """.format(req_data['access'], req_data['start_time'], req_data['endtime'],
        req_data['place'], req_data['zone'], req_data['videostream'],
        req_data['objective'], req_data['active'])

    # execute the query (function returns id of the new row)
    id = db_execute_query(query)

    # schedule detection if active == 1
    if req_data['active'] == 1:
        s = sched.scheduler(time.time, time.sleep)
        s.enter(req_data['start_time'] - int(time.time()), 0, detection, kwargs={'id': id, 'endtime': req_data['endtime']})
        t = Thread(target=s.run)
        t.start()
    elif req_data['active'] == 1 and req_data['start_time'] == 0:
        t = Thread(target=detection, args=[id, 0])
        t.start()

    return 'success'


def send_notifier(obj):
    print('=======', obj)
    # api_url = 'http://google.com'
    # r = requests.post(url=api_url, data=obj)
    # print(r.status_code, r.reason, r.text)


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=30, threaded=True)