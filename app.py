import os, time, sched, json, requests, datetime
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
            stream_url='/stream/' + str(id),
            target='stream',
            id=str(id))
    else:
        # create preview folder and add preview.jpeg there
        if not os.path.exists(os.getcwd() + '/data/previews/' + str(id)+ '.jpeg'):
            copyfile(os.getcwd() + '/data/images/preview.jpeg',
                    os.getcwd() + '/data/previews/' + str(id) + '.jpeg')
        # Show zone preview page.
        return render_template('index.html',
            stream_url='/preview/' + str(id),
            target='preview',
            id=str(id))


@app.route('/<id>/zone')
def zone_selection(id):
    if not id.isdigit():
        return '404'

    if not db_task_info(id):
        return 'no task planned'

    if os.path.exists(os.getcwd() + '/detections/' + str(id)):
        # Video streaming home page.
        return render_template('index.html',
            stream_url='/photo/' + str(id),
            target='photo',
            id=str(id))
    else:
        # create preview folder and add preview.jpeg there
        if not os.path.exists(os.getcwd() + '/data/previews/' + str(id)+ '.jpeg'):
            copyfile(os.getcwd() + '/data/images/preview.jpeg',
                    os.getcwd() + '/data/previews/' + str(id) + '.jpeg')
        # Show zone preview page.
        return render_template('index.html',
            stream_url='/preview/' + str(id),
            target='preview',
            id=str(id))


@app.route('/<id>/preview')
def preview_page(id):
    # create preview folder and add preview.jpeg there
    if not os.path.exists(os.getcwd() + '/data/previews/' + str(id)+ '.jpeg'):
        copyfile(os.getcwd() + '/data/images/preview.jpeg',
                os.getcwd() + '/data/previews/' + str(id) + '.jpeg')
    # Show zone preview page.
    return render_template('index.html',
        stream_url='/preview/' + str(id),
        target='preview',
        id=str(id))


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


def last_frame(camera, id):
    frame = camera.get_frame(id)
    yield (b'--frame\r\n'
           b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


@app.route('/stream/<id>')
def stream(id):
    """Video streaming route"""
    return Response(gen(Camera(), id),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/photo/<id>')
def photo(id):
    """Video streaming route"""
    return Response(last_frame(Camera(), id),
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
          videostream, videostreamid, regulationid, iddepartment,
          objective, bodyguard, active, pointx, pointy, width, height)
        VALUES
          (\'{}\', {}, {}, \'{}\', \'{}\', \'{}\', \'{}\', \'{}\', \'{}\', \'{}\', \'{}\', \'{}\', {}, {}, {}, {}, {});
    """.format(req_data['access'], req_data['start_time'], req_data['endtime'],
        req_data['place'], req_data['controlplace'], req_data['zone'], req_data['activezone'],
        req_data['videostream'], req_data['videostreamid'], req_data['regulationid'], req_data['idDepartment'],
        req_data['objective'], json.dumps(req_data['bodyguard']), req_data['active'],
        req_data['pointx'], req_data['pointy'], req_data['width'], req_data['height'])

    # execute the query (function returns id of the new row)
    id = db_execute_query(query)

    if req_data['pointx'] != 0 and req_data['width'] != 0:
        image = cv2.imread(os.getcwd() + '/data/images/preview.jpeg')
        image = highlight_zone(image, req_data['pointx'], req_data['pointy'], req_data['width'], req_data['height'])
        cv2.imwrite(os.getcwd() + '/data/previews/' + str(id) + '.jpeg',
                    image, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

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
        SET activezone = 1, pointx = {}, pointy = {}, width = {}, height = {}
        WHERE id = {}
    """.format(x, y, width, height, stream_id)
    db_execute_query(query)

    image = cv2.imread(os.getcwd() + '/data/images/preview.jpeg')
    image = highlight_zone(image, x, y, width, height)
    cv2.imwrite(os.getcwd() + '/data/previews/' + str(stream_id) + '.jpeg',
                image, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

    stream_info = db_task_info(stream_id)
    idVideostream, idPlace, zone = stream_info[9], stream_info[4], stream_info[6]

    # send updated zone to 1C
    url="https://copub.1c-ksu.ru/GPN_20/ws/ExchangeVideoserverAreas/ExchangeVideoserverAreas.1cws"
    #headers = {'content-type': 'application/soap+xml'}
    headers = {'content-type': 'text/xml'}
    body = """
    <soap:Envelope xmlns:soap="http://www.w3.org/2003/05/soap-envelope" xmlns:c="http://www.1c.exchange-videoserver-areas.serv.org" xmlns:c1="http://www.1c.exchange-videoserver-areas.org">
    <soap:Header/>
       <soap:Body>
          <c:addVSareas>
             <c:metadata>
                <!--Optional:-->
                <c1:idRequest>idRequest_2</c1:idRequest>
             </c:metadata>
             <c:data>
                <c1:pointX>{}</c1:pointX>
                <c1:pointY>{}</c1:pointY>
                <c1:width>{}</c1:width>
                <c1:height>{}</c1:height>
                <c1:idVideostream>{}</c1:idVideostream>
                <c1:idPlace>{}</c1:idPlace>
                <!--Optional:-->
                <c1:zone>{}</c1:zone>
             </c:data>
          </c:addVSareas>
       </soap:Body>
    </soap:Envelope>""".format(x, y, width, height, idVideostream, idPlace, zone)
    body = body.encode(encoding='utf-8')

    response = requests.post(url, data=body, headers=headers, auth=('WebServerVideo', 'Videoanalitika2020'), verify=False)
    print('sending zone coords result: ' + str(response.text))

    return 'success\n'


def send_notifier(stream_id, base64image):
    stream_info = db_task_info(stream_id)
    idAccess, idVideostream, idRegulation, idOrganization, idDepartment, idObjective = stream_info[1], stream_info[9], stream_info[10], stream_info[11], stream_info[12], stream_info[13]
    idAccess = "00000000-0000-0000-0000-000000000000" if idAccess == "0" else idAccess
    now = datetime.now()
    dateOffense = now.strftime("%Y-%m-%d") + "T" + now.strftime("%H:%M:%S")

    # send violated image to 1C
    url="https://copub.1c-ksu.ru/GPN_20/ws/ExchangeVideoserverOffences/ExchangeVideoserverOffences.1cws"
    #headers = {'content-type': 'application/soap+xml'}
    headers = {'content-type': 'text/xml'}
    body = """
    <soap:Envelope xmlns:soap="http://www.w3.org/2003/05/soap-envelope" xmlns:c="http://www.1c.exchange-videoserver-offences.serv.org" xmlns:c1="http://www.1c.exchange-videoserver-offences.org">
       <soap:Header/>
       <soap:Body>
          <c:addVSoffence>
             <c:metadata>
                <!--Optional:-->
                <c1:idRequest>idRequest_5</c1:idRequest>
             </c:metadata>
             <c:data>
                <!--Optional:-->
                <c1:idAccess>{}</c1:idAccess>
                <c1:idVideostream>{}</c1:idVideostream>
                <c1:idObjective>{}</c1:idObjective>
                <c1:idRegulation>{}</c1:idRegulation>
                <!--Optional:-->
                <c1:idOrganization>{}</c1:idOrganization>
                <!--Optional:-->
                <c1:idDepartment>{}</c1:idDepartment>
                <c1:dateOffense>{}</c1:dateOffense>
                <c1:image>{}</c1:image>
            </c:data>
          </c:addVSoffence>
       </soap:Body>
    </soap:Envelope>""".format(idAccess, idVideostream, idObjective, idRegulation, idOrganization, idDepartment, dateOffense, base64image)
    body = body.encode(encoding='utf-8')

    response = requests.post(url, data=body, headers=headers, auth=('WebServerVideo', 'Videoanalitika2020'), verify=False)
    print('sending violated image result: ' + str(response.text))

    return 'success\n'



if __name__ == '__main__':
    app.run(host=ip, port=port, threaded=True)
