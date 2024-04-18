import os
import argparse
import cv2
import numpy as np
import sys
import time
from threading import Thread

# Define VideoStream class to handle streaming of video from webcam in separate processing thread
class VideoStream:
    def __init__(self, src=0, resolution=(640,480), framerate=30):
        self.stream = cv2.VideoCapture(src)
        ret = self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        ret = self.stream.set(3,resolution[0])
        ret = self.stream.set(4,resolution[1])   
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False

    def start(self):
        Thread(target=self.update,args=()).start()
        return self

    def update(self):
        while True:
            if self.stopped:
                self.stream.release()
                return
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True

# Define and parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--modeldir', help='Folder the .tflite file is located in', required=True)
parser.add_argument('--graph', help='Name of the .tflite file, if different than detect.tflite', default='detect.tflite')
parser.add_argument('--labels', help='Name of the labelmap file, if different than labelmap.txt', default='labelmap.txt')
parser.add_argument('--threshold', help='Minimum confidence threshold for displaying detected objects', default=0.5)
parser.add_argument('--resolution', help='Desired webcam resolution in WxH. If the webcam does not support the resolution entered, errors may occur.', default='1280x720')
parser.add_argument('--edgetpu', help='Use Coral Edge TPU Accelerator to speed up detection', action='store_true')

args = parser.parse_args()

MODEL_NAME = args.modeldir
GRAPH_NAME = args.graph
LABELMAP_NAME = args.labels
min_conf_threshold = float(args.threshold)
resW, resH = map(int, args.resolution.split('x'))  # Parse resolution width and height as integers
use_TPU = args.edgetpu

# Importing Interpreter class from tflite_runtime if available, otherwise from tensorflow.lite.python
try:
    from tflite_runtime.interpreter import Interpreter
    from tflite_runtime.interpreter import load_delegate
except ImportError:
    from tensorflow.lite.python.interpreter import Interpreter
    from tensorflow.lite.python.interpreter import load_delegate

# If using Edge TPU, assign filename for Edge TPU model
if use_TPU:
    if GRAPH_NAME == 'detect.tflite':
        GRAPH_NAME = 'edgetpu.tflite'       

CWD_PATH = os.getcwd()
PATH_TO_CKPT = os.path.join(CWD_PATH, MODEL_NAME, GRAPH_NAME)
PATH_TO_LABELS = os.path.join(CWD_PATH, MODEL_NAME, LABELMAP_NAME)

with open(PATH_TO_LABELS, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

if labels[0] == '???':
    del(labels[0])

# Load the Tensorflow Lite model
if use_TPU:
    interpreter = Interpreter(model_path=PATH_TO_CKPT, experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
else:
    interpreter = Interpreter(model_path=PATH_TO_CKPT)

interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]
floating_model = (input_details[0]['dtype'] == np.float32)
input_mean = 127.5
input_std = 127.5

outname = output_details[0]['name']
if 'StatefulPartitionedCall' in outname:
    boxes_idx, classes_idx, scores_idx = 1, 3, 0
else:
    boxes_idx, classes_idx, scores_idx = 0, 1, 2

frame_rate_calc = 1
freq = cv2.getTickFrequency()

# Predefined box coordinates for camera 1 with numbers and distance thresholds
predefined_boxes_cam1 = [
    ((291,187,329,240), 'Box 1', 7),
    ((504,300,539,334), 'Box 2', 9),
    ((763,295,798,321), 'Box 3', 9),
    ((763,330,795,353), 'Box 4', 12),
    ((897,191,920,230), 'Box 5', 10)
]

# Predefined box coordinates for camera 2 with numbers and distance thresholds
predefined_boxes_cam2 = [
    ((100,100,150,150), 'Box 1', 7),
    ((200,200,250,250), 'Box 2', 9),
    ((300,300,350,350), 'Box 3', 9),
    ((400,400,450,450), 'Box 4', 12),
    ((500,500,550,550), 'Box 5', 10)
]

# Create instances of VideoStream for both cameras
videostream1 = VideoStream(src=0, resolution=(resW, resH), framerate=30).start()
videostream2 = VideoStream(src=4, resolution=(resW, resH), framerate=30).start()
time.sleep(1)

while True:
    t1 = cv2.getTickCount()
    frame1 = videostream1.read()
    frame2 = videostream2.read()
    
    # Perform object detection on frame1
    frame1_rgb = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
    frame1_resized = cv2.resize(frame1_rgb, (width, height))
    input_data1 = np.expand_dims(frame1_resized, axis=0)

    if floating_model:
        input_data1 = (np.float32(input_data1) - input_mean) / input_std

    interpreter.set_tensor(input_details[0]['index'], input_data1)
    interpreter.invoke()

    boxes1 = interpreter.get_tensor(output_details[boxes_idx]['index'])[0]
    classes1 = interpreter.get_tensor(output_details[classes_idx]['index'])[0]
    scores1 = interpreter.get_tensor(output_details[scores_idx]['index'])[0]

    for i in range(len(scores1)):
        if ((scores1[i] > min_conf_threshold) and (scores1[i] <= 1.0)):
            ymin = int(max(1,(boxes1[i][0] * resH)))
            xmin = int(max(1,(boxes1[i][1] * resW)))
            ymax = int(min(resH,(boxes1[i][2] * resH)))
            xmax = int(min(resW,(boxes1[i][3] * resW)))

            cv2.rectangle(frame1, (xmin, ymin), (xmax, ymax), (10, 255, 0), 2)

            object_name = labels[int(classes1[i])]
            label = '%s: %d%%' % (object_name, int(scores1[i]*100))
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            label_ymin = max(ymin, labelSize[1] + 10)
            cv2.rectangle(frame1, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED)
            cv2.putText(frame1, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

            # Check for intersection with predefined boxes for camera 1
            for box_coords, box_number, distance_threshold in predefined_boxes_cam1:
                box_xmin, box_ymin, box_xmax, box_ymax = box_coords
                if xmin < box_xmax and xmax > box_xmin and ymin < box_ymax and ymax > box_ymin:
                    # Calculate distance from camera (you need to implement this)
                    distance_from_camera = 10  # Implement this function
                    if distance_from_camera > distance_threshold:
                        # Display notification with box number
                        notification = 'Packman Captures %s!' % box_number
                        notificationSize, _ = cv2.getTextSize(notification, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                        cv2.rectangle(frame1, (xmin, ymax), (xmin+notificationSize[0], ymax+notificationSize[1]+10), (0, 255, 255), cv2.FILLED)
                        cv2.putText(frame1, notification, (xmin, ymax+notificationSize[1]+5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    # Drawing predefined boxes for camera 1
    for box_coords, _, _ in predefined_boxes_cam1:
        cv2.rectangle(frame1, (box_coords[0], box_coords[1]), (box_coords[2], box_coords[3]), (0, 0, 255), 2)

    # Perform object detection on frame2
    frame2_rgb = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
    frame2_resized = cv2.resize(frame2_rgb, (width, height))
    input_data2 = np.expand_dims(frame2_resized, axis=0)

    if floating_model:
        input_data2 = (np.float32(input_data2) - input_mean) / input_std

    interpreter.set_tensor(input_details[0]['index'], input_data2)
    interpreter.invoke()

    boxes2 = interpreter.get_tensor(output_details[boxes_idx]['index'])[0]
    classes2 = interpreter.get_tensor(output_details[classes_idx]['index'])[0]
    scores2 = interpreter.get_tensor(output_details[scores_idx]['index'])[0]

    for i in range(len(scores2)):
        if ((scores2[i] > min_conf_threshold) and (scores2[i] <= 1.0)):
            ymin = int(max(1,(boxes2[i][0] * resH)))
            xmin = int(max(1,(boxes2[i][1] * resW)))
            ymax = int(min(resH,(boxes2[i][2] * resH)))
            xmax = int(min(resW,(boxes2[i][3] * resW)))

            cv2.rectangle(frame2, (xmin, ymin), (xmax, ymax), (10, 255, 0), 2)

            object_name = labels[int(classes2[i])]
            label = '%s: %d%%' % (object_name, int(scores2[i]*100))
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            label_ymin = max(ymin, labelSize[1] + 10)
            cv2.rectangle(frame2, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED)
            cv2.putText(frame2, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

            # Check for intersection with predefined boxes for camera 2
            for box_coords, box_number, distance_threshold in predefined_boxes_cam2:
                box_xmin, box_ymin, box_xmax, box_ymax = box_coords
                if xmin < box_xmax and xmax > box_xmin and ymin < box_ymax and ymax > box_ymin:
                    # Calculate distance from camera (you need to implement this)
                    distance_from_camera = 10  # Implement this function
                    if distance_from_camera > distance_threshold:
                        # Display notification with box number
                        notification = 'Packman Captures %s!' % box_number
                        notificationSize, _ = cv2.getTextSize(notification, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                        cv2.rectangle(frame2, (xmin, ymax), (xmin+notificationSize[0], ymax+notificationSize[1]+10), (0, 255, 255), cv2.FILLED)
                        cv2.putText(frame2, notification, (xmin, ymax+notificationSize[1]+5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    # Drawing predefined boxes for camera 2
    for box_coords, _, _ in predefined_boxes_cam2:
        cv2.rectangle(frame2, (box_coords[0], box_coords[1]), (box_coords[2], box_coords[3]), (0, 0, 255), 2)

    # Display frames
    cv2.imshow('Camera 1', frame1)
    cv2.imshow('Camera 2', frame2)

    if cv2.waitKey(1) == ord('q'):
        break

cv2.destroyAllWindows()
videostream1.stop()
videostream2.stop()
