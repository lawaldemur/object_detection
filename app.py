import os, time, sched, json, requests
from importlib import import_module
from shutil import copyfile
from flask import Flask, render_template, Response, request
from threading import Thread
from camera import Camera
from db_connection import *
from detect import *


# local: for mac os 0.0.0.0:5001, for windows 127.0.0.1:30
ip, port = '0.0.0.0', 5001
app = Flask(__name__)

@app.route('/')
def index():
    # Video streaming home page.
    return 'main page'

@app.route('/<id>')
def stream_page(id):
    if not id.isdigit():
        return '404'

    if not db_task_info(id):
        return 'no task planned'

    if os.path.exists(os.getcwd() + '/detections/' + str(id)):
        # Video streaming home page.
        return render_template('index.html',
            stream_url='/stream/' + request.path[1:],
            id=request.path[1:])
    else:
        # create preview folder and add preview.jpeg there
        if not os.path.exists(os.getcwd() + '/data/previews/' + str(id)+ '.jpeg'):
            copyfile(os.getcwd() + '/data/images/preview.jpeg',
                    os.getcwd() + '/data/previews/' + str(id) + '.jpeg')
        # Show zone preview page.
        return render_template('index.html',
            stream_url='/preview/' + request.path[1:],
            id=request.path[1:])


def gen(camera, id):
    """Video streaming generator function."""
    while True:
        frame = camera.get_frame(id)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

def static(camera, path):
    frame = camera.get_static(path)
    yield (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


@app.route('/stream/<id>')
def stream(id):
    """Video streaming route"""
    return Response(gen(Camera(), id),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/preview/<id>')
def preview(id):
    """Video streaming route"""
    return Response(static(Camera(), os.getcwd() + '/data/previews/' + str(id) + '.jpeg'),
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

    return 'http://' + ip + ':' + str(port) + '/' + str(id)


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
        UPDATE requests_log
        SET activezone = 1, x = {}, y = {}, width = {}, height = {}
        WHERE id = {}
    """.format(x, y, width, height, stream_id)
    db_execute_query(query)

    image = cv2.imread(os.getcwd() + '/data/images/preview.jpeg')
    image = highlight_zone(image, x, y, width, height)
    cv2.imwrite(os.getcwd() + '/data/previews/' + str(stream_id) + '.jpeg',
                image, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

    # send updated zone to 1C
    url="https://db.1c-ksu.ru/VA_Prombez2/ws/ExchangeVideoserverPoints/ExchangeVideoserverPoints.1cws?wsdl"
    #headers = {'content-type': 'application/soap+xml'}
    headers = {'content-type': 'text/xml'}
    body = """
    <soapenv:Envelope xmlns:soapenv="http://schemas.xmlsoap.org/soap/envelope/" xmlns:c="http://www.1c.exchange-videoserver-points.serv.org" xmlns:c1="http://www.1c.exchange-videoserver-points.org">
       <soapenv:Header/>
       <soapenv:Body>
          <c:addVSpoints>
             <c:metadata>
                <!--Optional:-->
                <c1:idRequest>86032790-f41c-11ea-a41e-4cedfb43b7af</c1:idRequest>
             </c:metadata>
             <c:data>
                <c1:pointX>{}</c1:pointX>
                <c1:pointY>{}</c1:pointY>
                <c1:width>{}</c1:width>
                <c1:height>{}</c1:height>
                <c1:idVideostream>86032790-f41c-11ea-a41e-4cedfb43b7af</c1:idVideostream>
                <c1:idObjective>86032790-f41c-11ea-a41e-4cedfb43b7af</c1:idObjective>
                <c1:zone>Холл в центре завода</c1:zone>
             </c:data>
          </c:addVSpoints>
       </soapenv:Body>
    </soapenv:Envelope>""".format(x, y, width, height)
    body = """
    <soap:Envelope xmlns:soap="http://www.w3.org/2003/05/soap-envelope" xmlns:c="http://www.1c.exchange-videoserver-points.serv.org" xmlns:c1="http://www.1c.exchange-videoserver-points.org">
       <soap:Header/>
       <soap:Body>
          <c:addVSpoints>
             <c:metadata>
                <!--Optional:-->
                <c1:idRequest>idRequest</c1:idRequest>
             </c:metadata>
             <c:data>
                <c1:pointX>{}</c1:pointX>
                <c1:pointY>{}</c1:pointY>
                <c1:width>{}</c1:width>
                <c1:height>{}</c1:height>
                <c1:idVideostream>{}</c1:idVideostream>
                <c1:idObjective>0</c1:idObjective>
                <c1:zone>0</c1:zone>
             </c:data>
          </c:addVSpoints>
       </soap:Body>
    </soap:Envelope>""".format(x, y, width, height, stream_id)
    body = body.encode(encoding='utf-8')

    print("ready to send post")
    response = requests.post(url, data=body, headers=headers, auth=('WebServerVideo', 'Videoanalitika2020'))
    print('post result: ' + str(response))

    return 'success\n'


def send_notifier(obj):
    print('=======', obj)
    # api_url = 'http://google.com'
    # r = requests.post(url=api_url, data=obj)
    # print(r.status_code, r.reason, r.text)


if __name__ == '__main__':
    app.run(host=ip, port=port, threaded=True)
