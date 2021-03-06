import cv2
import time
import numpy as np
import os

CONFIDENCE_THRESHOLD = 0.6
NMS_THRESHOLD = 0.001
item_list = ["blur", "covered", "empty", "iron", "redsand", "stone_and_whitesand", "tanker", "white_sand"]

cap = cv2.VideoCapture('rtsp://admin:JnLuA2iF2N@115.243.39.105:554/ch1/stream3')
prevTime = 0
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#cv2.namedWindow("output", cv2.WINDOW_NORMAL)    

class_names = []
with open("anpr.names", "r") as f:
    class_names = [cname.strip() for cname in f.readlines()]

    
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
size = (frame_width, frame_height)


colors = np.random.uniform(0,255,size=(len(class_names),3))
net = cv2.dnn.readNet("anpr.weights", "anpr.cfg")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)

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
        color=colors[classid]
        label = "%s : %f" % (class_names[classid], score)
        cv2.rectangle(frame, box,color, 2)
        cv2.putText(frame, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,255), 2)
        if class_names[classid] in item_list:
            ctr += 1
            if ctr%3==0:
                if name%10000==0:
                    dir_n += 1
                    if not os.path.exists("/home/pilot/External_Storage/anpr/new" + str(dir_n)):
                        path = "/home/pilot/External_Storage/anprnew" + str(dir_n)
                        os.mkdir(path)
                f_name = "./new"+str(dir_n)+"/"+str(name)+".jpg"
                cv2.imwrite(f_name, frame)
                print(name)
                name += 1