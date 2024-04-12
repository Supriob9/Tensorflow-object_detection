import os
import argparse
import cv2
import numpy as np
import sys
import time
from threading import Thread
import importlib.util

# Define VideoStream class to handle streaming of video from webcam in separate processing thread
class VideoStream:
    def __init__(self,resolution=(640,480),framerate=30):
        self.stream = cv2.VideoCapture(0)
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

# Import TensorFlow libraries
try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    from tensorflow.lite.python.interpreter import Interpreter

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
    from tflite_runtime.interpreter import load_delegate
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

# Define VideoStream object
videostream = VideoStream(resolution=(resW, resH), framerate=30).start()
time.sleep(1)

# Predefined box coordinates with numbers
predefined_boxes = [
    ((291,187,329,240), 'Box 1'),
    ((504,300,539,334), 'Box 2'),
    ((763,295,798,321), 'Box 3'),
    ((763,330,795,353), 'Box 4'),
    ((897,191,920,230), 'Box 5')
]

# Define cube box parameters
cube_color = (0, 0, 255)  # Red color for cube
cube_thickness = 2  # Thickness of cube lines
box_distance = 20  # Assumed distance between original and cubic boxes

# Copy predefined boxes and calculate corners of cubic boxes
cubic_boxes = []

for box_coords, _ in predefined_boxes:
    box_xmin, box_ymin, box_xmax, box_ymax = box_coords
    cubic_box_coords = (box_xmin - box_distance, box_ymin - box_distance, box_xmax - box_distance, box_ymax - box_distance)
    cubic_boxes.append((box_coords, cubic_box_coords))

while True:
    t1 = cv2.getTickCount()
    frame1 = videostream.read()
    frame = frame1.copy()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (width, height))
    input_data = np.expand_dims(frame_resized, axis=0)

    if floating_model:
        input_data = (np.float32(input_data) - input_mean) / input_std

    interpreter.set_tensor(input_details[0]['index'],input_data)
    interpreter.invoke()

    boxes = interpreter.get_tensor(output_details[boxes_idx]['index'])[0]
    classes = interpreter.get_tensor(output_details[classes_idx]['index'])[0]
    scores = interpreter.get_tensor(output_details[scores_idx]['index'])[0]

    for i in range(len(scores)):
        if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)):
            ymin = int(max(1,(boxes[i][0] * resH)))
            xmin = int(max(1,(boxes[i][1] * resW)))
            ymax = int(min(resH,(boxes[i][2] * resH)))
            xmax = int(min(resW,(boxes[i][3] * resW)))
            
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (10, 255, 0), 2)

            object_name = labels[int(classes[i])]
            label = '%s: %d%%' % (object_name, int(scores[i]*100))
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            label_ymin = max(ymin, labelSize[1] + 10)
            cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED)
            cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

            # Check for intersection with predefined cube boxes
            for original_coords, cubic_coords in cubic_boxes:
                original_xmin, original_ymin, original_xmax, original_ymax = original_coords
                cubic_xmin, cubic_ymin, cubic_xmax, cubic_ymax = cubic_coords
                
                # Define intersection coordinates
                intersection_xmin = max(original_xmin, cubic_xmin)
                intersection_ymin = max(original_ymin, cubic_ymin)
                intersection_xmax = min(original_xmax, cubic_xmax)
                intersection_ymax = min(original_ymax, cubic_ymax)

               # print("Intersection:", (intersection_xmin, intersection_ymin, intersection_xmax, intersection_ymax))
                #print("Object Box:", (xmin, ymin, xmax, ymax))
                #print("Original Box:", original_coords)
                #print("Cubic Box:", cubic_coords)

                if xmin < intersection_xmax and xmax > intersection_xmin and ymin < intersection_ymax and ymax > intersection_ymin:
                 #   print("Intersection Found!")
                    # Get the box number associated with the current predefined box
                    for box_num, coords in enumerate(predefined_boxes):
                        if coords == original_coords:
                            box_number = box_num + 1
                            # Display notification with box number
                            notification = 'Packman Captures Box %s!' % box_number
                            notificationSize, _ = cv2.getTextSize(notification, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                            print("Notification:", notification)
                            cv2.rectangle(frame, (xmin, ymax), (xmin+notificationSize[0], ymax+notificationSize[1]+10), (0, 255, 255), cv2.FILLED)
                            cv2.putText(frame, notification, (xmin, ymax+notificationSize[1]+5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                       

    # Drawing original and cubic boxes along with connection lines
    for original_coords, cubic_coords in cubic_boxes:
        original_xmin, original_ymin, original_xmax, original_ymax = original_coords
        cubic_xmin, cubic_ymin, cubic_xmax, cubic_ymax = cubic_coords

        # Draw original and cubic boxes
        cv2.rectangle(frame, (original_xmin, original_ymin), (original_xmax, original_ymax), cube_color, cube_thickness)
        cv2.rectangle(frame, (cubic_xmin, cubic_ymin), (cubic_xmax, cubic_ymax), cube_color, cube_thickness)

        # Draw lines to connect corners of original and cubic boxes
        for i in range(4):
            cv2.line(frame, (original_xmin, original_ymin), (cubic_xmin, cubic_ymin), cube_color, cube_thickness)
            cv2.line(frame, (original_xmax, original_ymin), (cubic_xmax, cubic_ymin), cube_color, cube_thickness)
            cv2.line(frame, (original_xmin, original_ymax), (cubic_xmin, cubic_ymax), cube_color, cube_thickness)
            cv2.line(frame, (original_xmax, original_ymax), (cubic_xmax, cubic_ymax), cube_color, cube_thickness)

    cv2.imshow('Object detector', frame)

    t2 = cv2.getTickCount()
    time1 = (t2-t1)/freq
    frame_rate_calc = 1/time1

    if cv2.waitKey(1) == ord('q'):
        break

cv2.destroyAllWindows()
videostream.stop()
