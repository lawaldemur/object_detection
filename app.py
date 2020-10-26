import os, time, sched, json, requests
from importlib import import_module
from flask import Flask, render_template, Response, request
from threading import Thread
from camera import Camera
from db_connection import *
from detect import *


app = Flask(__name__)


@app.route('/')
def index():
    # Video streaming home page.
    return 'main page'

@app.route('/<id>')
def stream_page(id):
    # Video streaming home page.
    return render_template('index.html', path=request.path[1:])


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
          requests_log (access, start_time, endtime,
          place, controlplace, zone, activezone,
          videostream, videostreamid, regulationid,
          objective, bodyguard, active)
        VALUES
          (\'{}\', {}, {}, \'{}\', \'{}\', \'{}\', \'{}\', \'{}\', \'{}\', \'{}\', \'{}\', \'{}\', {});
    """.format(req_data['access'], req_data['start_time'], req_data['endtime'],
        req_data['place'], req_data['controlplace'], req_data['zone'], req_data['activezone'],
        req_data['videostream'], req_data['videostreamid'], req_data['regulationid'],
        req_data['objective'], json.dumps(req_data['bodyguard']), req_data['active'])

    # execute the query (function returns id of the new row)
    id = db_execute_query(query)

    # start detection immediately if start_time less than now or schedule detection
    if req_data['active'] == 1 and req_data['start_time'] <= int(time.time()):
        t = Thread(target=detection, args=[id, req_data['endtime']])
        t.start()
    elif req_data['active'] == 1:
        s = sched.scheduler(time.time, time.sleep)
        s.enter(req_data['start_time'] - int(time.time()), 0, detection, kwargs={'id': id, 'endtime': req_data['endtime']})
        t = Thread(target=s.run)
        t.start()

    return 'success\n'


@app.route('/change_zone')
def change_coords():
    # receive data in json format
    stream_id = request.args.get('stream_id', 0, type=int)
    x = request.args.get('x', 0, type=int)
    y = request.args.get('y', 0, type=int)
    width = request.args.get('width', 0, type=int)
    height = request.args.get('height', 0, type=int)

    # add info about new zone to database
    query = """
        INSERT INTO
          zone_coords (stream_id, x, y, width, height, timestamp)
        VALUES
          ({}, {}, {}, {}, {}, {});
    """.format(stream_id, x, y, width, height, int(time.time()))
    db_execute_query(query)

    return 'success\n'


def send_notifier(obj):
    print('=======', obj)
    # api_url = 'http://google.com'
    # r = requests.post(url=api_url, data=obj)
    # print(r.status_code, r.reason, r.text)


if __name__ == '__main__':
	# local: for mac os 0.0.0.0:5000, for windows 127.0.0.1:30
    app.run(host='0.0.0.0', port=5001, threaded=True)
