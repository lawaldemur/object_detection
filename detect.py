import os, time, glob, base64, json
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from absl import flags, logging
from absl.flags import FLAGS
import core.utils as utils
from core.yolov4 import filter_boxes
from core.functions import count_objects
from tensorflow.python.saved_model import tag_constants
from PIL import Image
import cv2
import numpy as np
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from app import send_notifier
from db_connection import *
from matplotlib import pyplot as plt


with open(os.getcwd() + '/config.json') as json_config:
    config = json.load(json_config)['detection']

# main vars
framework = 'tf'
size = 416
tiny = False
model = 'yolov4'
output_format = 'XVID'

iou = 0.5
score_human = 0.76
score_obj = 0.88
count = False
dont_show = True
info = False
# skip = 29
skip = 40
show_fps = False
outline = False
violation_threshold = 0.5
check_in_frames = (5 * 30) // (skip + 1)
# last_boxes_period = ((60 // (skip + 1)) + 1)
# last_boxes = []


weights = os.getcwd() + config['people_model']
video = os.getcwd() + config['detecting_video']
detection_folder = config['detection_folder']
output = config['output']
output_path = os.getcwd() + config['output_path']
save_last_frame = config['last_frame']
last_frame_name = config['last_frame_name']
equipments_names = config['equipments_names']

print('start loading models...')
# main models (for detecting persons)
saved_model_loaded = tf.saved_model.load(weights, tags=[tag_constants.SERVING])
infer = saved_model_loaded.signatures['serving_default']
# additional equipment models
infers = {}
for key, value in config['models'].items():
    infers[key] = tf.saved_model.load(os.getcwd() + value, tags=[tag_constants.SERVING])
    # infers[key] = infers[key].signatures['serving_default']
print('models loaded')


def detect_on_person(original_image, bodyguard):
    # original_image = cv2.imread(original_image)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    input_size = size

    image_data = cv2.resize(original_image, (input_size, input_size))
    image_data = image_data / 255.
    image_data = image_data[np.newaxis, ...].astype(np.float32)

    batch_data = tf.constant(image_data)
    output = []

    for equip in bodyguard:
        infer2 = infers[equip].signatures['serving_default']

        pred_bbox = infer2(batch_data)
        for key, value in pred_bbox.items():
            boxes = value[:, :, 0:4]
            pred_conf = value[:, :, 4:]

        # run non max suppression on detections
        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=iou,
            score_threshold=0
        )

        output.append([valid_detections.numpy()[0], classes.numpy()[0], scores.numpy()[0], equipments_names[equip]])
        
    return output


def get_zone_of_image(image, x, y, w, h):
    return image[y:y+h, x:x+w]

def put_image_on_image(image, res, x, y, w, h):
    image[y:y+h, x:x+w] = res
    return image

def check_not_empty_zone_coords(z):
    return not(len(z[0]) == 0)


def highlight_zone(image, x, y, w, h, color=(102, 255, 255)):
    # First we crop the sub-rect from the image
    sub_img = get_zone_of_image(image, x, y, w, h)
    white_rect = np.full(sub_img.shape, color, dtype=np.uint8)
    res = cv2.addWeighted(sub_img, 0.5, white_rect, 0.5, 1.0)
    # Putting the image back to its position
    image[y:y+h, x:x+w] = res

    return image


def get_detected_zone(result_frame, bodyguard=['helmet'], forbidden=False):
    input_size = size
    frame_size = result_frame.shape[:2]
    image_data = cv2.resize(result_frame, (input_size, input_size))
    image_data = image_data / 255.
    image_data = image_data[np.newaxis, ...].astype(np.float32)
    # detect on full image or part of image
    batch_data = tf.constant(image_data)


    pred_bbox = infer(batch_data)
    for key, value in pred_bbox.items():
        boxes = value[:, :, 0:4]
        pred_conf = value[:, :, 4:]

    boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
        boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
        scores=tf.reshape(
            pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
        max_output_size_per_class=50,
        max_total_size=50,
        iou_threshold=iou,
        score_threshold=score_human
    )

    # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, xmax, ymax
    original_h, original_w, _ = result_frame.shape
    bboxes = utils.format_boxes(boxes.numpy()[0], original_h, original_w)

    # if we control emptiness of a room
    if forbidden:
        return bboxes

    obj_detections = []

    image = Image.fromarray(result_frame)
    for i in range(valid_detections.numpy()[0]):
        # save persons parts
        image_tmp = image.crop((bboxes[i][0] - 10, bboxes[i][1] - 10, bboxes[i][2] + 10, bboxes[i][3] + 10))
        image_tmp = cv2.cvtColor(np.array(image_tmp), cv2.COLOR_BGR2RGB)

        obj_detections.append(detect_on_person(image_tmp, bodyguard))

    pred_bbox = [bboxes, scores.numpy()[0], classes.numpy()[0], valid_detections.numpy()[0]]
    image, violation = utils.draw_bbox(result_frame, pred_bbox, obj_detections, obj_threshold=score_obj)
    return image, violation


def detection(id, endtime, check_emptiness=False):
    print('Detection #', id, ' starts. Finish at ', endtime, sep='')

    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
    input_size = size
    video_path = video
    skip_frames = skip
    ended = False
    zone_coords = False
    frame_id = 0
    violations = []
    os.mkdir(os.getcwd() + detection_folder + '/' + str(id))

    # begin video capture
    try:
        vid = cv2.VideoCapture(int(video_path))
    except:
        vid = cv2.VideoCapture(video_path)

    out = None

    if output:
        # by default VideoCapture returns float instead of int
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*output_format)
        out = cv2.VideoWriter(output_path, codec, fps, (width, height))


    while True:
        if time.time() >= endtime and endtime > 0:
            print('Detection #{} is stopped due to endtime'.format(id))
            return

        for _ in range(skip_frames):
            vid.grab()

            
        return_value, frame = vid.read()
        if return_value:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            print('Video has ended')
            ended = True
            break

        start_time = time.time()
        notify = False


        if check_emptiness:
            if not is_empty(frame):
                print('ZONE IS NOT CLEAR!')
            # result = frame
            result = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            # get zone coords from database
            zones = db_task_info(id)[-4:]
            zone_coords = []
            forbidden_coords = []

            for i in range(len(zones)):
                z_coords = []
                f_coords = []

                for j in json.loads(zones[i]):
                    if j >= 0:
                        z_coords.append(j)
                    else:
                        f_coords.append(-j)

                zone_coords.append(z_coords)
                forbidden_coords.append(f_coords)


            bodyguard = json.loads(db_task_info(id)[14])
            if len(bodyguard) == 0:
                bodyguard = ['helmet']

            # for each zone
            if check_not_empty_zone_coords(zone_coords):
                # detect only inside of the zone
                # result_frame = get_zone_of_image(frame, zone_coords[0], zone_coords[1], zone_coords[2], zone_coords[3])
                for i in range(len(zone_coords[0])):
                    result_frame = get_zone_of_image(frame, zone_coords[0][i], zone_coords[1][i], zone_coords[2][i], zone_coords[3][i])
                    image, violation = get_detected_zone(result_frame, bodyguard)

                    frame = put_image_on_image(frame, image, zone_coords[0][i], zone_coords[1][i], zone_coords[2][i], zone_coords[3][i])
                    violations.append(violation)
            else:
                frame, violation = get_detected_zone(frame, bodyguard)
                violations.append(violation)
                
            image = frame
            
            result = np.asarray(image)
            result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

        
            if check_not_empty_zone_coords(zone_coords):
                # highlight zones
                for i in range(len(zone_coords[0])):
                    result = highlight_zone(result, zone_coords[0][i], zone_coords[1][i], zone_coords[2][i], zone_coords[3][i])
                # end for each zone
            if check_not_empty_zone_coords(forbidden_coords):
                # highlight zones
                for i in range(len(forbidden_coords[0])):
                    forbidden_zone = get_zone_of_image(result, forbidden_coords[0][i], forbidden_coords[1][i], forbidden_coords[2][i], forbidden_coords[3][i])
                    forbidden_people = get_detected_zone(forbidden_zone, forbidden=True)
                    for j in forbidden_people:
                        if (forbidden_coords[0][i] < j[0] and j[0] < (forbidden_coords[0][i] + forbidden_coords[2][i])) or \
                            (forbidden_coords[0][i] < j[2] and j[2] < (forbidden_coords[0][i] + forbidden_coords[2][i])) or \
                            (forbidden_coords[1][i] < j[1] and j[1] < (forbidden_coords[1][i] + forbidden_coords[3][i])) or \
                            (forbidden_coords[1][i] < j[3] and j[3] < (forbidden_coords[1][i] + forbidden_coords[3][i])):
                            print('MAN IN RED ZONE!!!')

                    result = highlight_zone(result, forbidden_coords[0][i], forbidden_coords[1][i], forbidden_coords[2][i], forbidden_coords[3][i], (0, 0, 195))

            # violaton sending
            while len(violations) > check_in_frames:
                del violations[0]


            if len(violations) == check_in_frames:
                avg_violation = sum([int(i) for i in violations]) / len(violations)
                if avg_violation > violation_threshold:
                    notify = True
                    violations.clear()


        if show_fps:
            fps = 1.0 / (time.time() - start_time)
            print("FPS: %.2f" % fps)

        if outline:
            result = cv2.Canny(result, 100, 200)

        if save_last_frame:
            # save as last deteciton
            cv2.imwrite(os.getcwd() + detection_folder + '/' + last_frame_name, result, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        
        # save to stream directory
        frame_id += 1
        cv2.imwrite(os.getcwd() + detection_folder + '/' + str(id) + '/' + str(frame_id).zfill(7) + '.jpeg', result, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

        if notify:
            tmp, buffer = cv2.imencode('.jpeg', result)
            send_notifier(id, base64.b64encode(buffer).decode('utf-8'))

        if output:
            out.write(result)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


def is_empty(result_frame):
    input_size = size
    frame_size = result_frame.shape[:2]
    image_data = cv2.resize(result_frame, (input_size, input_size))
    image_data = image_data / 255.
    image_data = image_data[np.newaxis, ...].astype(np.float32)
    # detect on full image or part of image
    batch_data = tf.constant(image_data)

    pred_bbox = infer(batch_data)
    for key, value in pred_bbox.items():
        boxes = value[:, :, 0:4]
        pred_conf = value[:, :, 4:]

    boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
        boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
        scores=tf.reshape(
            pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
        max_output_size_per_class=50,
        max_total_size=50,
        iou_threshold=iou,
        score_threshold=score_human
    )

    # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, xmax, ymax
    original_h, original_w, _ = result_frame.shape
    bboxes = utils.format_boxes(boxes.numpy()[0], original_h, original_w)

    summ_of_obj_probabilities = sum([sum(i) for i in bboxes])

    return summ_of_obj_probabilities == 0


def detect_on_single_image(path='/detections/sample.jpeg'):
    image = cv2.imread(os.getcwd() + path)
    image, violations = get_detected_zone(image)
    cv2.imwrite(os.getcwd() + path, image)


detect_on_single_image('/detections/01.jpg')
detect_on_single_image('/detections/02.jpg')
detect_on_single_image('/detections/03.jpg')
detect_on_single_image('/detections/04.jpg')
detect_on_single_image('/detections/05.jpg')
