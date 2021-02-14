import cv2
import imutils
from imutils.video import VideoStream

from tensorflow.python.keras.models import load_model

# load our face detector model
prototextPath = "/home/harshpatel/PycharmProjects/Project1/face_detector/deploy.prototxt"
weightsPath = "/home/harshpatel/PycharmProjects/Project1/face_detector/res10_300x300_ssd_iter_140000.caffemodel"
weightsPath = "/home/harshpatel/PycharmProjects/Project1/face_detector/res10_300x300_ssd_iter_140000.caffemodel"

facenet = cv2.dnn.readNet(prototextPath,weightsPath)

# load mask detector
masknet = load_model("mask_detector.model")

print("[Info] starting video Stream")

vs = VideoStream(src=0).start()


def detect_and_predict_mask(frame, facenet, masknet):
    print( frame.shape[:2])


while True:
    frame = vs.read()
    frame = imutils.resize(frame,width=400)

    detect_and_predict_mask(frame,facenet,masknet)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break