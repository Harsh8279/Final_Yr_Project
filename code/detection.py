import cv2
import imutils
from imutils.video import VideoStream
import numpy as np
from tensorflow.python.keras.applications.mobilenet_v2 import preprocess_input

from tensorflow.python.keras.models import load_model

# load our face detector model
from tensorflow.python.keras.preprocessing.image import img_to_array

prototextPath = "/home/harshpatel/PycharmProjects/Project1/face_detector/deploy.prototxt"
weightsPath = "/home/harshpatel/PycharmProjects/Project1/face_detector/res10_300x300_ssd_iter_140000.caffemodel"
weightsPath = "/home/harshpatel/PycharmProjects/Project1/face_detector/res10_300x300_ssd_iter_140000.caffemodel"

facenet = cv2.dnn.readNet(prototextPath,weightsPath)

# load mask detector
masknet = load_model("mask_detector.model")

print("[Info] starting video Stream")

vs = VideoStream(src=0).start()


def detect_and_predict_mask(frame, facenet, masknet):

    (h,w) = frame.shape[:2]     # frame height and width frame.shape = (300, 400, 3)

    blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),(104.0, 177.0, 123.0))

    '''
    blob==> 
    [[ -81.  -80.  -78. ...   16.   14.   15.]
   [ -81.  -81.  -79. ...   16.   15.   15.]
   [ -81.  -81.  -82. ...   14.   17.   16.]
   ...
   [-103. -103. -104. ...  -51.  -51.  -54.]
   [-106. -105. -105. ...  -52.  -52.  -53.]
   [-107. -105. -105. ...  -53.  -52.  -53.]]
    '''


    facenet.setInput(blob)


    detections = facenet.forward()
    #   facenet.forward ==> if we don't set the input then ==>    cv2.error: OpenCV(4.5.1) /tmp/pip-req-build-ms668fyv/opencv/modules/dnn/src/dnn.cpp:3108: error: (-215:Assertion failed) !layers[0].outputBlobs.empty() in function 'allocateLayers'

    '''
    after setting the input
       [[[[0.         1.         0.9999207  ... 0.4547274  0.56052285
       0.788671  ]
      [0.         1.         0.13460207 ... 4.0264554  5.1402316
       5.341313  ]
      [0.         1.         0.12179974 ... 4.026118   1.1148169
       5.343844  ]
      ...
      [0.         0.         0.         ... 0.         0.
       0.        ]
      [0.         0.         0.         ... 0.         0.
       0.        ]
      [0.         0.         0.         ... 0.         0.
       0.        ]]]]
       '''

    # print(detections.shape)  ==> (1, 1, 200, 7)

    faces = []
    locs = []
    preds = []

    for i in range(detections.shape[2]):
        confidence = detections[0,0,i,2]

        '''
        0.9987394
            0.12217447
            0.11303331
            0.108276926
            0.10363843
            0.098534696
            0.09785739
            0.09516203
            ....
            0.0
        0.9987155
            0.122624665
            0.11282766
            0.108533755
            0.10461074
            0.09822891
            0.09757469
            0.094179735
        '''

        if confidence>0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])     # [210.79502106 114.85360265 304.34060097 229.05730605]

            (startX,startY,endX,endY) = box.astype("int") # [210 114 304 229] etc ...

            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)

            faces.append(face)
            locs.append((startX, startY, endX, endY))

        if len(faces) > 0:
            # for faster inference we'll make batch predictions on *all*
            # faces at the same time rather than one-by-one predictions
            # in the above `for` loop
            faces = np.array(faces, dtype="float32")
            preds = masknet.predict(faces, batch_size=32)

        return (locs,preds)


while True:
    frame = vs.read()
    frame = imutils.resize(frame,width=400)

    (locs, preds) = detect_and_predict_mask(frame,facenet,masknet)

    for (box, pred) in zip(locs, preds):
        # unpack the bounding box and predictions
        (startX, startY, endX, endY) = box
        (mask, withoutMask) = pred

        # determine the class label and color we'll use to draw
        # the bounding box and text
        label = "Mask" if mask > withoutMask else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

        # include the probability in the label
        label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

        # display the label and bounding box rectangle on the output
        # frame
        cv2.putText(frame, label, (startX, startY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)



    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cv2.destroyAllWindows()
vs.stop()