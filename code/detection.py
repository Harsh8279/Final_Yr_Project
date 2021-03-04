import cv2
import imutils
from imutils.video import VideoStream
import numpy as np
from tensorflow.python.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.preprocessing.image import img_to_array
import os
import face_recognition
from datetime import datetime

# reco
def file_encoding(images):
    encode_list = []

    for i in images:
        i = cv2.cvtColor(i,cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(i)[0]
        encode_list.append(encode)

    return encode_list

# det
os.chdir("../face_detector")

# print(os.getcwd())


prototextPath = os.path.join(os.getcwd(), "deploy.prototxt")

# print(os.path.isfile(our_file))

exist_one = os.path.isfile(prototextPath)

if exist_one:
    weightsPath = os.path.join(os.getcwd(), "res10_300x300_ssd_iter_140000.caffemodel")

    exist_two = os.path.isfile(weightsPath)

    if exist_two:
        # print("hi!!!")
        os.chdir("../code")

        facenet = cv2.dnn.readNet(prototextPath, weightsPath)

        masknet = load_model("mask_detector.model")

        # reco
        path = "AttendenceImages"
        images = []
        classNames = []

        myList = os.listdir(path)

        for i in myList:
            curImg = cv2.imread(path + "/" + i)
            images.append(curImg)
            classNames.append(os.path.splitext(i)[0])

        encodeListKnown = file_encoding(images)

        print("Encoding Completed!!!")

        print("[Info] starting video Stream")

        # vs = VideoStream(src=0).start()
        vs = cv2.VideoCapture(0)

    else:
        print("caffe Model not found!!")
else:
    print("Prototext file not found")

#det
def detect_and_predict_mask(frame, facenet, masknet):
    (h, w) = frame.shape[:2]  # frame height and width frame.shape = (300, 400, 3)

    blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224), (104.0, 177.0, 123.0))

    facenet.setInput(blob)

    detections = facenet.forward()
    #   facenet.forward ==> if we don't set the input then ==>    cv2.error: OpenCV(4.5.1) /tmp/pip-req-build-ms668fyv/opencv/modules/dnn/src/dnn.cpp:3108: error: (-215:Assertion failed) !layers[0].outputBlobs.empty() in function 'allocateLayers'

    faces = []
    locs = []
    preds = []

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array(
                [w, h, w, h])  # [210.79502106 114.85360265 304.34060097 229.05730605]

            (startX, startY, endX, endY) = box.astype("int")  # [210 114 304 229] etc ...

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
            faces = np.array(faces, dtype="float32")
            preds = masknet.predict(faces, batch_size=32)

        return (locs, preds)


def face_mask_detetction_fun():
    # print(os.getcwd())

        while True:
            success,frame = vs.read()
            imgS = cv2.resize(frame, (0, 0), None, 0.25, 0.25)
            imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

            facesCurFrame = face_recognition.face_locations(imgS)
            encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

            for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
                matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
                faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)

                # print(faceDis)

                matchIndex = np.argmin(faceDis)

                if matches[matchIndex]:
                    name = classNames[matchIndex].upper()
                    markAttendence(name)

            (locs, preds) = detect_and_predict_mask(frame, facenet, masknet)

            for (box, pred) in zip(locs, preds):
                (startX, startY, endX, endY) = box
                (mask, withoutMask) = pred

                label = "Mask" if mask > withoutMask else "No Mask"
                color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

                label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

                cv2.putText(frame, label, (startX, startY - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

        cv2.destroyAllWindows()

        print(os.getcwd())





def markAttendence(name):
    with open('Attendence.csv','r+') as f:
        myDataList = f.readlines()

        # print(myDataList)
        nameList=[]
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H : %M : %S')
            f.writelines(f'\n{name},{dtString}')


# face_mask_detetction_fun()