# Import necessary packages
import os
import cv2
import numpy as np
import time
from threading import Thread
from tflite_runtime.interpreter import Interpreter

# Define VideoStream class
class VideoStream:
    """Camera object that controls video streaming from the Picamera"""
    def __init__(self,resolution=(640,480),framerate=30):
        # Initialize the PiCamera and the camera image stream
        self.stream = cv2.VideoCapture(0)
        ret = self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        ret = self.stream.set(3,resolution[0])
        ret = self.stream.set(4,resolution[1])
            
        # Read first frame from the stream
        (self.grabbed, self.frame) = self.stream.read()

        # Variable to control when the camera is stopped
        self.stopped = False

    def start(self):
        # Start the thread that reads frames from the video stream
        Thread(target=self.update,args=()).start()
        return self

    def update(self):
        # Keep looping indefinitely until the thread is stopped
        while True:
            # If the camera is stopped, stop the thread
            if self.stopped:
                # Close camera resources
                self.stream.release()
                return

            # Otherwise, grab the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        # Return the most recent frame
        return self.frame

    def stop(self):
        # Indicate that the camera and thread should be stopped
        self.stopped = True

# Define a function to check for intersection between two bounding boxes
def box_intersect(box1, box2):
    # Implementation to check for intersection between two boxes
    # Return True if there is an intersection, False otherwise
    return (box1[0] < box2[2] and box1[2] > box2[0] and box1[1] < box2[3] and box1[3] > box2[1])

# Define a function to check for intersection between probe and predefined boxes
def detect_intersection(probe_box, predefined_boxes):
    for box in predefined_boxes:
        if box_intersect(probe_box, box):
            return True
    return False

# Main function for object detection
def detect_objects(modeldir, graph, labels, threshold, resolution, predefined_boxes):
    # Get path to current working directory
    CWD_PATH = os.getcwd()

    # Path to .tflite file, which contains the model that is used for object detection
    PATH_TO_CKPT = os.path.join(CWD_PATH, modeldir, graph)

    # Load the Tensorflow Lite model
    interpreter = Interpreter(model_path=PATH_TO_CKPT)
    interpreter.allocate_tensors()

    # Get model details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]

    # Define input mean and std for normalization
    input_mean = 127.5
    input_std = 127.5

    # Initialize video stream
    videostream = VideoStream(resolution=(width, height), framerate=30).start()
    time.sleep(1)

    # Initialize variables for frame rate calculation
    frame_rate_calc = 1
    freq = cv2.getTickFrequency()

    # Main loop for object detection
    while True:
        # Start timer
        t1 = cv2.getTickCount()

        # Grab frame from video stream
        frame1 = videostream.read()

        # Acquire frame and resize to expected shape [1xHxWx3]
        frame = frame1.copy()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (width, height))
        input_data = np.expand_dims(frame_resized, axis=0)

        # Normalize pixel values if using a floating model
        if input_details[0]['dtype'] == np.float32:
            input_data = (np.float32(input_data) - input_mean) / input_std

        # Set input tensor to the image
        interpreter.set_tensor(input_details[0]['index'], input_data)

        # Run inference
        interpreter.invoke()

        # Retrieve detection results
        boxes = interpreter.get_tensor(output_details[0]['index'])[0]
        classes = interpreter.get_tensor(output_details[1]['index'])[0]
        scores = interpreter.get_tensor(output_details[2]['index'])[0]

        # Loop over all detections
        for i in range(len(scores)):
            if ((scores[i] > threshold) and (scores[i] <= 1.0)):
                # Get bounding box coordinates
                ymin, xmin, ymax, xmax = boxes[i]
                xmin = int(xmin * frame.shape[1])
                ymin = int(ymin * frame.shape[0])
                xmax = int(xmax * frame.shape[1])
                ymax = int(ymax * frame.shape[0])

                # Check for intersection with predefined boxes
                probe_box = (xmin, ymin, xmax, ymax)
                if detect_intersection(probe_box, predefined_boxes.get(image_filename, [])):
                    # Trigger notification function here
                    print("Intersection detected!")

                # Draw bounding box and label
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                label = "{}: {:.2f}%".format(labels[int(classes[i])], scores[i] * 100)
                cv2.putText(frame, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Draw predefined boxes on the webcam feed
        count = 0
        for image_filename, boxes in predefined_boxes.items():
            if count >= 5:
                break
            for box in boxes[:5]:  # Select the first 5 boxes
                xmin, ymin, xmax, ymax = box
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)  # Draw blue rectangle
                cv2.putText(frame, 'Predefined Box', (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)  # Add label
            count += 1

        # Display frame
        cv2.imshow('Object Detection', frame)

        # Calculate framerate
        t2 = cv2.getTickCount()
        time1 = (t2 - t1) / freq
        frame_rate_calc = 1 / time1

       # Check for quit command
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Stop the video stream when the loop ends
videostream.stop()

# Cleanup
cv2.destroyAllWindows()

# Define predefined boxes for each image
predefined_boxes = {

     'IMG20240306144822-OD8Xgcis-min.jpg': [(748, 642, 842, 770), (1393, 1084, 1507, 1167), (2293, 1155, 2407, 1238), (2356, 1019, 2456, 1090), (2745, 696, 2816, 824)]
}
