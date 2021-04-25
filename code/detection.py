import os
from datetime import datetime
from tkinter import *
from tkinter import ttk, filedialog

import cv2
import face_recognition
import imutils
import numpy as np
import pandas as pd
import serial
from PIL import Image, ImageTk
from scipy.spatial import distance as dist


# reco
def file_encoding(images):
    encode_list = []

    for i in images:
        i = cv2.cvtColor(i, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(i)[0]
        encode_list.append(encode)

    return encode_list

# reco
path = "./res/AttendenceImages"  # folder
images = []
classNames = []

myList = os.listdir(path)

for i in myList:
    curImg = cv2.imread(path + "/" + i)
    images.append(curImg)
    classNames.append(os.path.splitext(i)[0])

encodeListKnown = file_encoding(images)

print("Encoding Completed!!!")

def face_mask_detetction_fun():
    net = cv2.dnn.readNet('./res/yolov3_training_final.weights', 'yolov3_testing.cfg')  # file

    classes = []
    font = cv2.FONT_HERSHEY_DUPLEX
    with open('./res/classes.txt', 'r') as f:  # file
        classes = f.read().splitlines()

    # img = cv2.imread('test3.jpg')
    # print(img.shape)

    cap = cv2.VideoCapture(0)

    while True:
        success, img = cap.read()
        imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

        facesCurFrame = face_recognition.face_locations(imgS)
        encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)



        height, width, = img.shape[:2]

        colors = np.random.uniform(0, 255, size=(100, 3))

        blob = cv2.dnn.blobFromImage(img, 1 / 255, (416, 416), (0, 0, 0), swapRB=True, crop=False)
        net.setInput(blob)
        output_layers_names = net.getUnconnectedOutLayersNames()
        layerOutputs = net.forward(output_layers_names)

        boxes = []
        confidences = []
        class_ids = []

        for output in layerOutputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.2:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append((float(confidence)))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.2, 0.4)
        temp_func = temp_detection()
        if type(temp_func) == int:
            temp = temp_func
            status = "NORMAL"
        else:
            temp, status = temp_func
        name = "Not Known"
        if len(indexes) > 0:
            for i in indexes.flatten():
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                confidence = str(round(confidences[i], 2) * 100)
                for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
                    matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
                    faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)

                    matchIndex = np.argmin(faceDis)

                    if matches[matchIndex]:
                        name = classNames[matchIndex].upper()
                if label == "not wearing mask":
                    color = (0, 0, 255)

                else:
                    color = (0, 255, 0)
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                cv2.putText(img, label + " " + confidence + "%", (x, y + 20), font, 1, color, 2)
        cv2.imshow('Image', img)
        if (temp > 40) or label == "not wearing mask":
            markAttendence(name, temp, label, status)
        if cv2.waitKey(1) == 27:
            break
    # cv2.waitKey(0)

    cap.release()
    cv2.destroyAllWindows()


def temp_detection():
    arduino_port = "/dev/ttyUSB0"  # serial port of Arduino
    baud = 9600  # arduino uno runs at 9600 baud

    ser = serial.Serial(arduino_port, baud)
    # print("Connected to Arduino port:" + arduino_port)
    ser_bytes = ser.readline()
    try:
        decoded_bytes = int(float(ser_bytes[0:len(ser_bytes) - 2]))
        return decoded_bytes
    except:
        decoded_bytes = (ser_bytes[0:len(ser_bytes) - 2])
        lst = decoded_bytes.decode('UTF-8').split(',')
        decoded_bytes = int(float(lst[0]))
        status = lst[-1]
        return decoded_bytes, status

def append_data(lst):
    df = pd.DataFrame(lst)
    df.to_csv('./res/Attendence.csv', mode='a', index=False, header=False)  # file
    remove_duplicates()


def remove_duplicates():
    df = pd.read_csv('./res/Attendence.csv')  # file
    df.drop_duplicates(inplace=True)
    df.to_csv('./res/Attendence.csv', mode='w', index=False)


def markAttendence(name, temp, label, status):
    if label == "wearing mask":
        mask = "Wearing"
    else:
        mask = "Not Wearing"
    get_time = datetime.now().time().strftime("%H%M")  # now
    get_time = datetime.strptime(get_time, "%H%M").time()
    lst = {'Name': [name], 'Mask Status': [mask], 'Temperature': [temp], 'Date': [datetime.now().date()],
           'Time': [get_time], 'Temperature Status': [status]}
    append_data(lst)


# Social - distance Detection
def detect_people(frame, net, ln, personIdx=0):
    (H, W) = frame.shape[:2]  # get dimensions of the frame
    results = []

    # construct a blob from input frame and perform yolo object detector, that give us boundary boxes and associated proprbability
    blob = cv2.dnn.blobFromImage(frame, 1 / 255, (416, 416), (0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)
    layerOutputs = net.forward(ln)

    boxes = []
    confidences = []
    centroids = []

    for output in layerOutputs:  # abstract all info from layeroutputs
        for detection in output:  # abstract the each info from outputs
            scores = detection[5:]  # abstract the classid and its confidence
            classId = np.argmax(scores)
            confidence = scores[classId]

            if classId == personIdx and confidence > 0.5:  # check for person object and mimmum confidence
                box = detection[0:4] * np.array(
                    [W, H, W, H])  # scale bounding box cordinates back relative to the size of image
                (centerX, centerY, width, height) = box.astype("int")

                # get top left cordinates of the frame(img/object)
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                boxes.append([x, y, int(width), int(height)])
                centroids.append((centerX, centerY))
                confidences.append(float(confidence))

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)  # 0.5 - min conficence         0.3 - NMS threshold

    if len(idxs) > 0:
        for i in idxs.flatten():
            # extract the bounding box cordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            # update the result with person detection probability
            r = (confidences[i], (x, y, x + w, y + h), centroids[i])
            results.append(r)

    return results


def social_distance_detection():
    net = cv2.dnn.readNet('./res/yolov3.weights', './res/yolov3.cfg')  # file

    classes = []  # LABELS
    with open('./res/coco.names', 'r') as f:  # file
        classes = f.read().splitlines()

    # print(classes)

    # determine only the output layer that we need from yolo
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    # img = cv2.imread('1.jpg')
    cap = cv2.VideoCapture("./res/test.mp4")  # video                 # file
    # cap = cv2.VideoCapture(0)       # live web cam

    writer = None

    while True:

        (grabbed, frame) = cap.read()

        if not grabbed:
            break

        frame = imutils.resize(frame, width=700)
        results = detect_people(frame, net, ln, personIdx=classes.index("person"))

        violate = set()

        if len(results) >= 2:
            #     extract all centroids from result and calculate euclidean distance between them
            centroids = np.array([r[2] for r in results])
            D = dist.cdist(centroids, centroids, metric="euclidean")

            # loop over the upper triangular of the distance matrix
            for i in range(0, D.shape[0]):
                for j in range(i + 1, D.shape[1]):
                    if D[i, j] < 75:  # 75 - min distance
                        violate.add(i)
                        violate.add(j)

        for (i, (prob, bbox, centroid)) in enumerate(results):
            (startX, startY, endX, endY) = bbox
            (cX, cY) = centroid
            color = (0, 255, 0)

            if i in violate:
                color = (0, 0, 255)

            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
            cv2.circle(frame, (cX, cY), 5, color, 1)

        text = "Social Distancing Violations: {}".format(len(violate))
        cv2.putText(frame, text, (10, frame.shape[0] - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 3)

        cv2.imshow("img", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


# Only Face Mask Detection
def only_mask_detection():
    net = cv2.dnn.readNet('./res/yolov3_training_final.weights', './res/yolov3_testing.cfg')  # file

    classes = []
    font = cv2.FONT_HERSHEY_DUPLEX
    with open('./res/classes.txt', 'r') as f:  # file
        classes = f.read().splitlines()

    # img = cv2.imread('test3.jpg')
    # print(img.shape)

    cap = cv2.VideoCapture(0)

    while True:
        success, img = cap.read()
        height, width, = img.shape[:2]

        colors = np.random.uniform(0, 255, size=(100, 3))

        blob = cv2.dnn.blobFromImage(img, 1 / 255, (416, 416), (0, 0, 0), swapRB=True, crop=False)
        net.setInput(blob)
        output_layers_names = net.getUnconnectedOutLayersNames()
        layerOutputs = net.forward(output_layers_names)

        boxes = []
        confidences = []
        class_ids = []

        for output in layerOutputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.2:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append((float(confidence)))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.2, 0.4)

        if len(indexes) > 0:
            for i in indexes.flatten():
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                confidence = str(round(confidences[i], 2) * 100)
                if label == "not wearing mask":
                    color = (0, 0, 255)
                else:
                    color = (0, 255, 0)
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                cv2.putText(img, label + " " + confidence + "%", (x, y + 20), font, 1, color, 2)
        cv2.imshow('Image', img)
        if cv2.waitKey(1) == 27:
            break
    # cv2.waitKey(0)

    cap.release()
    cv2.destroyAllWindows()


root = Tk()

root['background'] = '#82eefd'

image = Image.open(r"./res/TestOne.jpg")  # file

image = image.resize((700, 300), Image.ANTIALIAS)

root.title("Face Mask Detection")

frame1 = Frame(root)
frame2 = Frame(root)
frame3 = Frame(root, borderwidth=6, relief=GROOVE)

frame1.pack(side=TOP)
frame1.place(x=20, y=40)
frame2.pack(side=BOTTOM)
frame2.place(x=750, y=80)
frame3.pack(side=BOTTOM)
frame3.place(x=50, y=420)

frame2['background'] = '#82eefd'
frame1['background'] = '#82eefd'


def file_open():
    filename = filedialog.askopenfilename(
        initialdir=os.getcwd(),
        title="Open A file",
        filetypes=(("csv files", "*.csv"), ("all files", "*.*"))
    )

    if filename:
        try:
            filename = r"{}".format(filename)
            df = pd.read_csv(filename)

            # clear the Tree view
            clear_tree()

            # set up new Tree view
            my_tree["column"] = list(df.columns)
            my_tree["show"] = "headings"

            # loop thru column list

            for column in my_tree["column"]:
                my_tree.heading(column, text=column)

            df_rows = df.to_numpy().tolist()

            for row in df_rows:
                my_tree.insert("", "end", values=row)

            my_tree.grid(row=0, column=0)
            btn_close.grid(row=1, column=0)

        except ValueError:
            my_label.config(text="File couldn't be open")

        except FileNotFoundError:
            my_label.config(text="File Not Found !!")


def clear_tree():
    my_tree.delete(*my_tree.get_children())

label = Label(frame1)
# label_two = Label(frame1)

photo = ImageTk.PhotoImage(image)
# photo_two = ImageTk.PhotoImage(image_two)

label['image'] = photo
# label_two['image'] = photo_two

label.grid(row=0, column=0, pady=5)

btn_face_mask_detection = Button(frame2, text="Face Mask Detection", command=only_mask_detection)
btn_face_mask_detection.grid(row=0, column=0, padx=20, pady=20)

btn_social_distance = Button(frame2, text="Social Distance Detection", command=social_distance_detection)
btn_social_distance.grid(row=0, column=1, padx=20, pady=20)

btn_face_mask_recognition = Button(frame2, text="Face Mask And Temperature Detection", command=face_mask_detetction_fun)
btn_face_mask_recognition.grid(row=1, column=0, padx=20, pady=20)

# btn_mask_temperature_detection = Button(frame2, text="Face Mask and Temperature Detection")
# btn_mask_temperature_detection.grid(row=1, column=1, padx=20, pady=20)

btn_open = Button(frame2, text="Open", command=file_open)
btn_open.grid(row=1, column=1, padx=20, pady=20)

btn_close = Button(frame3, text="Close", command=clear_tree)

my_tree = ttk.Treeview(frame3)

my_label = Label(root, text='')
my_label.pack(pady=20)

root.geometry("1500x800")

root.mainloop()