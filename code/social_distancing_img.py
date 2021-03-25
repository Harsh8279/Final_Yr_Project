import cv2
import numpy as np
import imutils
from scipy.spatial import distance as dist

def detect_people(frame, net, ln, personIdx=0):
    (H,W) = frame.shape[:2]    # get dimensions of the frame
    results = []
    
    # construct a blob from input frame and perform yolo object detector, that give us boundary boxes and associated proprbability
    blob = cv2.dnn.blobFromImage(frame, 1/255, (416,416), (0,0,0), swapRB=True, crop=False)
    net.setInput(blob)
    layerOutputs = net.forward(ln)
    
    boxes = []
    confidences = []
    centroids = []
    
    for output in layerOutputs:  #abstract all info from layeroutputs
        for detection in output:  # abstract the each info from outputs
            scores = detection[5:]  # abstract the classid and its confidence
            classId = np.argmax(scores)
            confidence = scores[classId]

            if classId == personIdx and confidence > 0.5:  # check for person object and mimmum confidence
                box = detection[0:4] * np.array([W, H, W, H])  # scale bounding box cordinates back relative to the size of image
                (centerX, centerY, width, height) = box.astype("int")

                # get top left cordinates of the frame(img/object)
                x = int(centerX - (width/2))
                y = int(centerY - (height/2))

                boxes.append([x, y, int(width), int(height)])
                centroids.append((centerX, centerY))
                confidences.append(float(confidence))

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)   # 0.5 - min conficence         0.3 - NMS threshold
    
    if len(idxs) > 0:
        for i in idxs.flatten():
            # extract the bounding box cordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            
            
            # update the result with person detection probability
            r = (confidences[i], (x, y, x+w, y+h), centroids[i])
            results.append(r)
            
    return results

net = cv2.dnn.readNet('yolov3.weights','yolov3.cfg')

classes = []  #LABELS
with open ('coco.names','r') as f:
    classes = f.read().splitlines()
    
# print(classes)

# determine only the output layer that we need from yolo 
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

img = cv2.imread('1.jpg')

height, width, _ = img.shape

frame = imutils.resize(img, width=700)
results = detect_people(frame, net, ln, personIdx = classes.index("person"))

violate = set()



if len(results) >=2:
#     extract all centroids from result and calculate euclidean distance between them
    centroids = np.array([r[2] for r in results])
    D = dist.cdist(centroids, centroids, metric="euclidean")
    
    
    # loop over the upper triangular of the distance matrix
    for i in range(0, D.shape[0]):
        for j in range(i+1, D.shape[1]):
            if D[i,j] < 75:             # 75 - min distance
                violate.add(i)
                violate.add(j)
                
                
for(i, (prob, bbox, centroid)) in enumerate(results):
    (startX, startY, endX, endY) = bbox
    (cX, cY) = centroid
    color = (0, 255, 0)
    
    if i in violate:
        color = (0 , 0, 255)
        
    cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
    cv2.circle(frame, (cX, cY), 5, color, 1)
    
text = "Social Distancing Violations: {}".format(len(violate))
cv2.putText(frame, text, (10, frame.shape[0] - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 3)


cv2.imshow("img",frame)
cv2.waitKey(0)
cv2.destroyAllWindows()



