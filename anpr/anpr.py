import cv2
import time
import numpy as np
import os

# from google.cloud import vision
# os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r'creds.json'
# client = vision.ImageAnnotatorClient()

CONFIDENCE_THRESHOLD = 0.6
NMS_THRESHOLD = 0.1

cap = cv2.VideoCapture('rtsp://root:P5FvY7hTyN@192.168.150.104:554/live1s1.sdp')

class_names = []
with open("mineral.names", "r") as f:
    class_names = [cname.strip() for cname in f.readlines()]

colors = np.random.uniform(0,255,size=(len(class_names),3))
net = cv2.dnn.readNet("anpr.weights", "anpr.cfg")
model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(640, 640), scale=1/255, swapRB=True)

name = 1
ctr = 0
dir_n = 1
while cap.isOpened(): 
    ret, frame = cap.read()
    frame = np.array(frame)
    print(name+1)
    
    classes, scores, boxes = model.detect(frame, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
    
    for (classid, score, box) in zip(classes, scores, boxes):
            ctr += 1
            if ctr%3==0:
                if name%10000==0:
                    dir_n += 1
                    if not os.path.exists("/home/pilot/External_Storage/mineral/new" + str(dir_n)):
                        path = "/home/pilot/External_Storage/mineral/new" + str(dir_n)
                        os.mkdir(path)
                f_name = "./new"+str(dir_n)+"/"+str(name)+".jpg"
                cv2.imwrite(f_name, frame)
                print(name)
                name += 1