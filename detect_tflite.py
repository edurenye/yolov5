from absl import app, flags, logging
from absl.flags import FLAGS

from fuzzy_alert import FuzzyDangerDetector
from model_tflite import load_model
import cv2
import numpy as np
import math
import time

flags.DEFINE_string('classes', './data/_classes.txt', 'path to classes file')
flags.DEFINE_string('model_type', 'pytorch', 'model type')
flags.DEFINE_string('model', '', 'path to model file')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_string('image', '', 'path to input image')
flags.DEFINE_string('video', '', 'path to input image')
flags.DEFINE_string('output', '', 'path to output image')
flags.DEFINE_string('output_format', 'MJPG', 'codec used in VideoWriter when saving video to file')


def main(_argv):
    model = load_model(app, model_type=FLAGS.model_type, model_path=FLAGS.model, classes_path=FLAGS.classes)
    logging.info('OpenCV version: ' + cv2.__version__)

    if FLAGS.image:
        img_raw = cv2.imread(FLAGS.image)
        img_raw = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)

        img, predictions = model.make_prediction(img_raw)
        print(predictions)

        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(FLAGS.output, img)
        logging.info('output saved to: {}'.format(FLAGS.output))
    
    if FLAGS.video:
        danger_detector = FuzzyDangerDetector()
        try:
            vid = cv2.VideoCapture(int(FLAGS.video))
        except:
            vid = cv2.VideoCapture(FLAGS.video)

        image_size = FLAGS.size
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        offset_w = int(round((width - image_size) / 2))
        offset_h = int(round((height - image_size) / 2))
        
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
        out = cv2.VideoWriter(FLAGS.output, codec, fps, (width, height))

        img = None
        counter = 0
        t1 = time.time()
        real_fps = 0
        current_objects = {
            'crosswalk': [],
            'dashed_crosswalk': [],
            'pedestrian_green': [],
            'pedestrian_off': [],
            'pedestrian_red': []
        }
        while True:
            _, img_raw = vid.read()

            if img_raw is None:
                logging.warning("Empty Frame")
                break
            logging.warning("Got one frame")

            img_raw = img_raw[offset_h:offset_h+image_size, offset_w:offset_w+image_size]

            img_raw = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)

            img, predictions = model.make_prediction(img_raw)

            update_current_objects(current_objects, predictions)
            dangers = get_dangers(danger_detector, current_objects)

            for danger in dangers:
                print(danger[2][0], danger[2][1])
                img = cv2.putText(img, str(danger[0]) + ' ' + str(danger[1]) + ' danger',
                                  (int(danger[2][0]), int(danger[2][1])),
                                  cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0, 0, 255), 1)

            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            img = cv2.putText(img, "FPS: {:.2f}".format(real_fps), (0, 30),
                              cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)

            counter += 1
            if counter > 10:
                t2 = time.time()
                spent_time = t2 - t1
                real_fps = float(counter) / spent_time
                print('FPS: ', real_fps)
                t1 = t2
                counter = 0

            out.write(img)

        logging.info('output saved to: {}'.format(FLAGS.output))
        out.release()


def update_current_objects(current_objects, predictions):
    now = time.time()
    for prediction in predictions:
        object_type = prediction[0]
        if object_type == 'traffic_light':
            continue
        else:
            found = False
            for current_object_type in current_objects:
                for current_object in current_objects[current_object_type]:
                    if are_the_same(current_object_type, current_object, prediction):
                        found = True
                        current_object['last_appearance'] = now
                        current_object['counter'] += 1
                        current_object['bounding_boxes'].append(prediction[1])
            if not found:
                current_objects[object_type].append({
                    'last_appearance': now,
                    'counter': 1,
                    'bounding_boxes': [prediction[1]]
                })
    for key in current_objects:
        for current_object in current_objects[key]:
            if current_object['last_appearance'] < (now - 2):
                current_objects[key].remove(current_object)


def are_the_same(current_object_type, current_object, prediction):
    pedestrian_tl = ['pedestrian_green', 'pedestrian_off', 'pedestrian_red']
    if current_object_type == prediction[0] or (current_object_type in pedestrian_tl and prediction[0] in pedestrian_tl):
        cx = current_object['bounding_boxes'][-1][0]
        cy = current_object['bounding_boxes'][-1][1]
        px = prediction[1][0]
        py = prediction[1][1]
        dist = math.sqrt((px - cx)**2 + (py - cy)**2)
        if dist < 20:
            return True
    return False


def get_dangers(danger_detector, current_objects):
    dangers = []
    for current_object_type in current_objects:
        for current_object in current_objects[current_object_type]:
            if current_object['counter'] > 10:
                growth = get_growth(current_object['bounding_boxes'])
                position = get_position(current_object['bounding_boxes'][-1])
                danger = danger_detector.get_danger(current_object_type, growth, position)
                if current_object_type == 'pedestrian_green':
                    danger = danger / 2
                print('Danger: ', current_object_type, danger)
                dangers.append([current_object_type, danger, current_object['bounding_boxes'][-1]])
    return dangers


def get_growth(bounding_boxes):
    first = bounding_boxes[0]
    last = bounding_boxes[-1]
    first_diagonal = math.sqrt(first[2] ** 2 + first[3] ** 2)
    last_diagonal = math.sqrt(last[2] ** 2 + last[3] ** 2)
    diagonal_increment = last_diagonal - first_diagonal
    ratio = (diagonal_increment / first_diagonal) * 40
    ratio = np.clip(ratio, 0, 10)
    print('Growth: ', ratio)
    return ratio


def get_position(xywh):
    center = xywh[0] + (float(xywh[2]) / 2)
    position = (center / FLAGS.size) * 10
    return position


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
