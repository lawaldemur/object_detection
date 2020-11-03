import os, time, glob, base64
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


# main vars
framework = 'tf'
weights = os.getcwd() + '/checkpoints/yolov4-416'
size = 416
tiny = False
model = 'yolov4'
video = os.getcwd() + '/data/video/ShortHelmets.mp4'
# output = os.getcwd() + '/detections/result.avi'
output = False
output_format = 'XVID'
save_last_frame = False
iou = 0.5
score_human = 0.76
# score_obj = 0.6
count = False
dont_show = True
info = False
skip = 29
show_fps = False
violation_threshold = 0.5
check_in_frames = (15 * 30) // (skip + 1)


print('start loading models...')
# main models (for detecting persons)
saved_model_loaded = tf.saved_model.load(weights, tags=[tag_constants.SERVING])
infer = saved_model_loaded.signatures['serving_default']
# additional models (only helmets yet)
additional_models = [tf.saved_model.load('./checkpoints/yolov4-helmet', tags=[tag_constants.SERVING])]
infer2 = additional_models[0].signatures['serving_default']
print('models loaded')


def detect_on_person(original_image):
    # original_image = cv2.imread(original_image)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    input_size = size

    image_data = cv2.resize(original_image, (input_size, input_size))
    image_data = image_data / 255.
    image_data = image_data[np.newaxis, ...].astype(np.float32)

    # images_data = []
    # for i in range(1):
    #     images_data.append(image_data)
    # images_data = np.asarray(images_data).astype(np.float32)

    batch_data = tf.constant(image_data)
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
    
    return [valid_detections.numpy()[0], classes.numpy()[0], scores.numpy()[0]]


def get_zone_of_image(image, x, y, w, h):
    return image[y:y+h, x:x+w]

def put_image_on_image(image, res, x, y, w, h):
    image[y:y+h, x:x+w] = res
    return image

def check_not_empty_zone_coords(z):
    return not(z[0] == 0 and z[1] == 0 and z[2] == 0 and z[3] == 0)


def highlight_zone(image, x, y, w, h, color=(102, 255, 255)):
    # First we crop the sub-rect from the image
    sub_img = get_zone_of_image(image, x, y, w, h)
    white_rect = np.full(sub_img.shape, color, dtype=np.uint8)
    res = cv2.addWeighted(sub_img, 0.5, white_rect, 0.5, 1.0)
    # Putting the image back to its position
    image[y:y+h, x:x+w] = res

    return image


def detection(id, endtime):
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
    os.mkdir(os.getcwd() + '/detections/' + str(id))

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
        out = cv2.VideoWriter(output, codec, fps, (width, height))


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

        # get zone coords from database
        zone_coords = db_task_info(id)[-4:]

        if check_not_empty_zone_coords(zone_coords):
            # detect only inside of the zone
            result_frame = get_zone_of_image(frame, zone_coords[0], zone_coords[1], zone_coords[2], zone_coords[3])
        else:
            result_frame = frame
            
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
        obj_detections = []

        image = Image.fromarray(result_frame)
        for i in range(valid_detections.numpy()[0]):
            # save persons parts
            image_tmp = image.crop((bboxes[i][0] - 10, bboxes[i][1] - 10, bboxes[i][2] + 10, bboxes[i][3] + 10))
            image_tmp = cv2.cvtColor(np.array(image_tmp), cv2.COLOR_BGR2RGB)

            obj_detections.append(detect_on_person(image_tmp) + [['КАСКА', 'НЕТ КАСКИ']])

        pred_bbox = [bboxes, scores.numpy()[0], classes.numpy()[0], valid_detections.numpy()[0]]
        image, violation = utils.draw_bbox(result_frame, pred_bbox, obj_detections, obj_threshold=0.86)

        violations.append(violation)

        # violaton sending
        # ((20 * 30) // (skip + 1)) + 1 equals 20 seconds of stream approximately 
        while len(violations) > check_in_frames:
            del violations[0]


        notify = False
        if len(violations) == check_in_frames:
            avg_violation = sum([int(i) for i in violations]) / len(violations)
            if avg_violation > violation_threshold:
                notify = True
                violations.clear()


        if show_fps:
            fps = 1.0 / (time.time() - start_time)
            print("FPS: %.2f" % fps)


        if check_not_empty_zone_coords(zone_coords):
            image = put_image_on_image(frame, result_frame, zone_coords[0], zone_coords[1], zone_coords[2], zone_coords[3])

        result = np.asarray(image)
        result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

        if check_not_empty_zone_coords(zone_coords):
            # highlight zone
            result = highlight_zone(result, zone_coords[0], zone_coords[1], zone_coords[2], zone_coords[3])


        if save_last_frame:
            # save as last deteciton
            cv2.imwrite(os.getcwd() + '/detections/last_frame.jpeg', result, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        
        # save to stream directory
        frame_id += 1
        cv2.imwrite(os.getcwd() + '/detections/' + str(id) + '/' + str(frame_id).zfill(7) + '.jpeg', result, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

        if notify:
            tmp, buffer = cv2.imencode('.jpeg', result)
            send_notifier(id, base64.b64encode(buffer))

        if output:
            out.write(result)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
