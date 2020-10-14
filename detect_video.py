import os, time, glob
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
iou = 0.45
score_human = 0.25
score_obj = 0.95
count = False
dont_show = True
info = False
skip = 0
show_fps = False
frame_id = 0
zone_highlighter = False


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

    images_data = []
    for i in range(1):
        images_data.append(image_data)
    images_data = np.asarray(images_data).astype(np.float32)

    batch_data = tf.constant(images_data)
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
        score_threshold=score_obj
    )

    for iscore in range(valid_detections.numpy()[0]):
        if scores.numpy()[0][iscore] < score_obj:
            send_notifier('no helmet ' + str(time.time()))
    
    return [valid_detections.numpy()[0], classes.numpy()[0], scores.numpy()[0]]


def highlight_zone(image, x, y, w, h, color=(102, 255, 255)):
    # First we crop the sub-rect from the image
    sub_img = image[y:y+h, x:x+w]
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
    frame_id = 0
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
            image = Image.fromarray(frame)
        else:
            print('Video has ended')
            ended = True
            break

        frame_size = frame.shape[:2]
        image_data = cv2.resize(frame, (input_size, input_size))
        image_data = image_data / 255.
        image_data = image_data[np.newaxis, ...].astype(np.float32)
        start_time = time.time()

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
        original_h, original_w, _ = frame.shape
        bboxes = utils.format_boxes(boxes.numpy()[0], original_h, original_w)
        obj_detections = []
        
        for i in range(valid_detections.numpy()[0]):
            # save persons parts
            image_tmp = image.crop((bboxes[i][0] - 10, bboxes[i][1] - 10, bboxes[i][2] + 10, bboxes[i][3] + 10))
            image_tmp = cv2.cvtColor(np.array(image_tmp), cv2.COLOR_BGR2RGB)

            obj_detections.append(detect_on_person(image_tmp) + [['КАСКА', 'НЕТ КАСКИ']])

        pred_bbox = [bboxes, scores.numpy()[0], classes.numpy()[0], valid_detections.numpy()[0]]
        image = utils.draw_bbox(frame, pred_bbox, obj_detections)

        if show_fps:
            fps = 1.0 / (time.time() - start_time)
            print("FPS: %.2f" % fps)

        result = np.asarray(image)
        result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)


        # highlight zone
        if zone_highlighter:
            result = highlight_zone(result, 1920 - 770, 0, 770, 1080)

        # save as last deteciton
        cv2.imwrite(os.getcwd() + '/detections/last_frame.jpeg', result, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        # save to stream directory
        frame_id += 1
        cv2.imwrite(os.getcwd() + '/detections/' + str(id) + '/' + str(frame_id).zfill(7) + '.jpeg', result, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

        if output:
            out.write(result)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
