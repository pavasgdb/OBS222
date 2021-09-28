import os
import cv2
import time
import argparse
import numpy as np
import subprocess as sp
import json
import tensorflow as tf
import serial
import matplotlib.pyplot as plt
import math

from queue import Queue
from threading import Thread
from utils.app_utils import FPS, HLSVideoStream, WebcamVideoStream, draw_boxes_and_labels
from object_detection.utils import label_map_util

(width , height, ll) = (0,0,0)
zz = 0 
z = 0
zz1 = 11
nex = str('a')
banex = bytes(nex, encoding="ascii")
ang = 90
y1 = 1
y2 = 0
l = 1

# defining function to be used as calibration: we would try to directly relate the gun angle with the pixel coordinate
# below are the points that must be used for calibration. In this we will try to relate the angle with the x-pixel-coordinate
points1 = np.array([(6.52, 0.5) , (56,4.5) ,(92.3,7.5)]) # write three values for three different angles as Y
points2 = np.array([(12,0.5) , (13,4.5) , (16,7.5)])

# get x and y vectors
x11 = points1[:,0]
y11 = points1[:,1]

# calculate polynomial
z1 = np.polyfit(x11, y11, 1)
f1 = np.poly1d(z)
#now poly is ready to be used
x22 = points2[:,0]
y22 = points2[:,1]

# calculate polynomial
z2 = np.polyfit(x22, y22, 1)
f2 = np.poly1d(z2)

ser = serial.Serial("/dev/tty.usbmodem14101",9600)
print("Trying to connect")
connected = False
while not connected:
    serin = ser.read()
    connected = True
# making sure that it is connected and changing the value of connected to true
print(serin)
print("Connected with Arduino")
#l = ser.read()
#print(l)

CWD_PATH = os.getcwd()

# Path to frozen detection graph. This is the actual model that is used for the object detection.
MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'
PATH_TO_CKPT = os.path.join(CWD_PATH, 'object_detection', MODEL_NAME, 'frozen_inference_graph.pb')

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join(CWD_PATH, 'object_detection', 'data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90

# Loading label map
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)
k2 = 3


def detect_objects(image_np, sess, detection_graph):
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

    # Each box represents a part of the image where a particular object was detected.
    boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    scores = detection_graph.get_tensor_by_name('detection_scores:0')
    classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    # Actual detection.
    (boxes, scores, classes, num_detections) = sess.run(
        [boxes, scores, classes, num_detections],
        feed_dict={image_tensor: image_np_expanded})
    
    # defining the values of width and height of image
    global width
    global height
    global ll
    
    (width , height, ll) = image_np.shape
     
    # Visualization of the results of a detection.
    rect_points, class_names, class_colors = draw_boxes_and_labels(
        boxes=np.squeeze(boxes),
        classes=np.squeeze(classes).astype(np.int32),
        scores=np.squeeze(scores),
        category_index=category_index,
        min_score_thresh=.5
    )
    return dict(rect_points=rect_points, class_names=class_names, class_colors=class_colors)
    #return (width , height , ll)

#print(width ,height ,ll)

def worker(input_q, output_q):
    # Load a (frozen) Tensorflow model into memory.
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

        sess = tf.Session(graph=detection_graph)

    fps = FPS().start()
    while True:
        fps.update()
        frame = input_q.get()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        output_q.put(detect_objects(frame_rgb, sess, detection_graph))

    fps.stop()
    sess.close()


tt = time.time()
ii = 30
i = 30

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-strin', '--stream-input', dest="stream_in", action='store', type=str, default=None)
    parser.add_argument('-src', '--source', dest='video_source', type=int,
                        default=0, help='Device index of the camera.')
    parser.add_argument('-wd', '--width', dest='width', type=int,
                        default=640, help='Width of the frames in the video stream.')
    parser.add_argument('-ht', '--height', dest='height', type=int,
                        default=480, help='Height of the frames in the video stream.')
    parser.add_argument('-strout','--stream-output', dest="stream_out", help='The URL to send the livestreamed object detection to.')
    args = parser.parse_args()

    input_q = Queue(1)  # fps is better if queue is higher but then more lags
    output_q = Queue()
    for i in range(1):
        t = Thread(target=worker, args=(input_q, output_q))
        t.daemon = True
        t.start()

    if (args.stream_in):
        print('Reading from hls stream.')
        video_capture = HLSVideoStream(src=args.stream_in).start()
    else:
        print('Reading from webcam.')
        video_capture = WebcamVideoStream(src=args.video_source,
                                      width=args.width,
                                      height=args.height).start()
    fps = FPS().start()
    
    
    
    while True:
        frame = video_capture.read()
        input_q.put(frame)

        t = time.time()
        

        if output_q.empty():
            pass  # fill up queue
        else:
            font = cv2.FONT_HERSHEY_SIMPLEX
            data = output_q.get()
            rec_points = data['rect_points']
            class_names = data['class_names']
            #exploit this stuff
            #print(class_names)
            #print(rec_points)
            #time.sleep(2)  # delay for 5 seconds
            # now we will try to print the centre of the rectangle
            #print(rec_points[0]["xmin"])
            
            if len(rec_points) != 0:
                
                X = (0.5*(rec_points[0]["xmax"] + rec_points[0]["xmin"]))
                Y = (0.5*(rec_points[0]["ymax"] + rec_points[0]["ymin"]))
                # angle to be sent to arduino
                X1 = 100*(1 - X)
                #zz = int(X1*10)
                ##zz = int(X1*60)
                print(X1)
                if l == 1:
                    zz = int(10*f1(X1))
                if l == 2:
                    zz = int(10*f1(X1))
                
                zz = zz/10 # using poly to get the best angle 
                #zz = i
                if zz < 4 :
                    ang = 90 - (180/3.14)*math.atan((4-zz)/10)
                if zz > 4:
                    ang = 90 + (180/3.14)*math.atan((zz-4)/10)
                if zz == 4:
                    ang = 90
                
                if Y > 0.6:
                    y1 = 1
                if Y > 0.3 and Y < 0.6:
                    y1 = 2
                if Y < 0.3:
                    y1 = 2

                if y1 != y2:
                    z = str(int(999))
                    ba = bytes(z, encoding="ascii")
                    ser.write(ba)
                    print("Ready to Send Y signal")
                    ser.write(banex)
                    if Y > 0.6:
                        z = str(int(40))
                        ba = bytes(z, encoding="ascii")
                        ser.write(ba)
                        print("Y updating")
                        ser.write(banex)
                        y2 = 1
                    if Y > 0.3 and Y < 0.6:
                        z = str(int(50))
                        ba = bytes(z, encoding="ascii")
                        ser.write(ba)
                        print("Y updating")
                        ser.write(banex)
                        y2 = 2
                    if Y < 0.3:
                        z = str(int(60))
                        ba = bytes(z, encoding="ascii")
                        ser.write(ba)
                        print("Y updating")
                        ser.write(banex)
                        y2 = 3
                    z = str(int(9999))
                    ba = bytes(z, encoding="ascii")
                    ser.write(ba)
                    print("Y updated")
                    ser.write(banex)
            """
            else:
                z = str(int(8888))
                ba = bytes(z, encoding="ascii")
                ser.write(ba)
                ser.write(banex)
                print("Updating camera")
                #l = ser.read()
            """            
            #k2 = int(ser.read())
            #print("hello")
            #print(zz)
            #print(tt)
            # converting calbi to angle
            z = str(int(ang))
            ba = bytes(z, encoding="ascii")
            #print((X,Y))
            #print((X*width,Y*height))
            #k2 = 4
            """
            if (time.time()-tt) > 2:    
                ser.write(ba)
                print(ba)
                tt = time.time()
            """
            if z != zz1:
                ser.write(ba)
                print(int(ang))
                print(zz)
                ser.write(banex)
                #time.sleep(0.1)
            
            zz1 = z
            """
            zz1 = i
            i = i + 30
            if i == 180:
                i = 30
            """    
                #time.sleep(5)
            #if (class_names == [['person: 97%']]):
                #print('LOLOLOL')
            #print(k2)
            #k2 = int(ser.read())
            #print(ba)
            #print(k2)
            class_colors = data['class_colors']
            for point, name, color in zip(rec_points, class_names, class_colors):
                cv2.rectangle(frame, (int(point['xmin'] * args.width), int(point['ymin'] * args.height)),
                              (int(point['xmax'] * args.width), int(point['ymax'] * args.height)), color, 3)
                cv2.rectangle(frame, (int(point['xmin'] * args.width), int(point['ymin'] * args.height)),
                              (int(point['xmin'] * args.width) + len(name[0]) * 6,
                               int(point['ymin'] * args.height) - 10), color, -1, cv2.LINE_AA)
                cv2.putText(frame, name[0], (int(point['xmin'] * args.width), int(point['ymin'] * args.height)), font,
                            0.3, (0, 0, 0), 1)
            if args.stream_out:
                print('Streaming elsewhere!')
            else:
                cv2.imshow('Video', frame)

        fps.update()

        print('[INFO] elapsed time: {:.2f}'.format(time.time() - t))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    fps.stop()
    print('[INFO] elapsed time (total): {:.2f}'.format(fps.elapsed()))
    print('[INFO] approx. FPS: {:.2f}'.format(fps.fps()))

    video_capture.stop()
    cv2.destroyAllWindows()
