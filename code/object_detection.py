# -*- coding: utf-8 -*-
"""object_detection.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1taaR5sNJFvyxqr9HJ0U-v04TcDaOOYs5
"""

import cv2
import numpy as np

net = cv2.dnn.readNet('yolov3.weights','yolov3.cfg')

classes = []
with open ('coco.names','r') as f:
    classes = f.read().splitlines()
    
# print(classes)

img = cv2.imread('person.jpg')
height, width, _ = img.shape

blob = cv2.dnn.blobFromImage(img, 1/255, (416,416), (0,0,0), swapRB=True, crop=False)

net.setInput(blob)

output_layers_names = net.getUnconnectedOutLayersNames()
layerOutputs = net.forward(output_layers_names)

boxes = []
confidences = []
class_ids = []

for output in layerOutputs:  #abstract all info from layeroutputs
    for detection in output:  # abstract the each info from outputs
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        
        if confidence > 0.5:
            center_x = int(detection[0]*width)
            center_y = int(detection[1]*height)
            w = int(detection[2]*width)
            h = int(detection[3]*height)
            
            x = int(center_x - w/2)
            y = int(center_y - h/2)
            
            boxes.append([x, y, w, h])
            confidences.append((float(confidence)))
            class_ids.append(class_id)
        

indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)


font = cv2.FONT_HERSHEY_PLAIN
colors = np.random.uniform(0,255,size=(len(boxes), 3))

for i in indexes.flatten():
    x, y, w, h = boxes[i]
    label = str(classes[class_ids[i]])
    confidence = str(round(confidences[i],2))
    color = colors[i]
    cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
    cv2.putText(img, label + " " + confidence, (x, y+20), font, 2, (0, 0, 255), 2)
    

        
cv2.imshow('Image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()