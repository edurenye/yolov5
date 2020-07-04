from abc import ABCMeta

import cv2
import numpy as np
import onnx
import onnxruntime as nxrun
import sys
import time
import torch
import random

from utils.datasets import *
from utils.utils import *

yolo_max_boxes = 100
yolo_iou_threshold = 0.5
yolo_score_threshold = 0.5


def draw_outputs(img, outputs, class_names):
    boxes, objectness, classes, nums = outputs
    boxes, objectness, classes, nums = boxes[0], objectness[0], classes[0], nums[0]
    wh = np.flip(img.shape[0:2])
    for i in range(nums):
        x1y1 = tuple((np.array(boxes[i][0:2]) * wh).astype(np.int32))
        x2y2 = tuple((np.array(boxes[i][2:4]) * wh).astype(np.int32))
        img = cv2.rectangle(img, x1y1, x2y2, (255, 0, 0), 2)
        img = cv2.putText(img, '{} {:.4f}'.format(
            class_names[int(classes[i])], objectness[i]),
            x1y1, cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)
    return img


def load_model(app, model_type, model_path, classes_path, size = 416):
    return PigallModel.load_model(app, model_type, model_path, classes_path, size)


class PigallModel(metaclass=ABCMeta):

    def __init__(self, app, model_type, model_path, classes_path, size):
        self.app = app
        self.size = size
        self.model_type = model_type
        self.class_names = self.get_class_names(classes_path)

    @staticmethod
    def load_model(app, model_type, model_path, classes_path, size):
        if model_type == 'cv-tf':
            return CVTFPigallModel(app, model_type, model_path, classes_path, size)
        elif model_type == 'onnx':
            return OnnxPigallModel(app, model_type, model_path, classes_path, size)
        elif model_type == 'pytorch':
            return PyTorchPigallModel(app, model_type, model_path, classes_path, size)
        else:
            return None

    @staticmethod
    def get_class_names(classes_path):
        class_names = [c.strip() for c in open(classes_path).readlines()]
        print('We got ' + str(len(class_names)) + ' classes.', file=sys.stderr)
        return class_names


class CVTFPigallModel(PigallModel):

    def __init__(self, app, model_type, model_path, classes_path, size):
        super().__init__(app, model_type, model_path, classes_path, size)
        self.model = cv2.dnn.readNetFromTensorflow(model_path, './pbtxt/network.pbtxt')
        print('Model loaded.', file=sys.stderr)


class OnnxPigallModel(PigallModel):

    def __init__(self, app, model_type, model_path, classes_path, size):
        super().__init__(app, model_type, model_path, classes_path, size)
        model = onnx.load(model_path)
        onnx.checker.check_model(model)
        # self.model = cv2.dnn.readNetFromONNX(model_path)
        self.model = nxrun.InferenceSession(model_path)
        print('Model loaded.', file=sys.stderr)

    def make_prediction(self, inputs):
        dataset = LoadImages(inputs, img_size=self.size)
        img = None
        for path, img, im0s, vid_cap in dataset:
            print("The model expects input shape: ", self.model.get_inputs()[0].shape)
            print("The shape of the Image is: ", inputs.shape)
            input_name = self.model.get_inputs()[0].name
            t1 = time.time()
            img = self.model.run(None, {input_name: inputs})
            t2 = time.time()
            prediction_time = t2 - t1
            img = cv2.putText(img, "Time: {:.2f}ms".format(prediction_time * 1000), (0, 30),
                              cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (0, 0, 255), 2)
        return img


class PyTorchPigallModel(PigallModel):

    def __init__(self, app, model_type, model_path, classes_path, size):
        super().__init__(app, model_type, model_path, classes_path, size)
        self.device = torch_utils.select_device()
        self.half = self.device.type != 'cpu'
        self.model = torch.load(model_path, map_location=self.device)['model'].float()
        self.model.to(self.device).eval()
        if self.half:
            self.model.half()
        print('Model loaded.', file=sys.stderr)

    def make_prediction(self, inputs):
        # Get names and colors
        names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

        # Padded resize
        img0 = inputs
        img = letterbox(img0, new_shape=self.size)[0]

        # Convert
        img = img.transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        print("The shape of the Image is: ", img.shape)
        # Inference
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        pred = self.model(img)[0]

        # Apply NMS
        pred = non_max_suppression(pred, 0.4, 0.5)

        write_boxes = True
        predictions = []
        # Process detections
        for i, det in enumerate(pred):  # detections per image
            s= '%gx%g ' % img.shape[2:]
            gn = torch.tensor(img0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

                # Write results
                for *xyxy, conf, cls in det:
                    xywh = self.xyxy2xywh(torch.tensor(xyxy).tolist())
                    predictions.append([names[int(cls)], xywh, conf.item()])

                    if write_boxes:  # Add bbox to image
                        label = '%s %.2f' % (names[int(cls)], conf)
                        plot_one_box(xyxy, img0, label=label, color=colors[int(cls)], line_thickness=1)

        return img0, predictions

    @staticmethod
    def xyxy2xywh(xyxy):
        return [xyxy[0], xyxy[1], xyxy[2] - xyxy[0], xyxy[3] - xyxy[1]]
