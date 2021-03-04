import cv2
import numpy as np
from tensorflow.python.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.preprocessing.image import img_to_array
import os
import face_recognition
from datetime import datetime
from tkinter import *
from PIL import Image,ImageTk
from tkinter import ttk,filedialog
import pandas as pd

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

        # print(os.getcwd())

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

root = Tk()

image =Image.open(r"TestOne.jpg")
# image_two =Image.open(r"recoOne.jpg")

image = image.resize((800, 350), Image.ANTIALIAS)
# image_two = image_two.resize((430, 350), Image.ANTIALIAS)

root.title("Face Mask Detection")

frame1 = Frame(root)
frame2 = Frame(root)
frame3 = Frame(root,borderwidth=6,relief=GROOVE)

frame1.pack(side=TOP)
frame1.place(x=20,y=40)
frame2.pack(side=BOTTOM)
frame2.place(x=250,y=440)
frame3.pack(side=BOTTOM)
frame3.place(x=870,y=400)


def file_open():
    filename = filedialog.askopenfilename(
        initialdir= os.getcwd(),
        title= "Open A file",
        filetypes=(("csv files","*.csv"),("all files","*.*"))
    )

    if filename:
        try:
            filename=r"{}".format(filename)
            df = pd.read_csv(filename)

        except ValueError:
            my_label.config(text="File couldn't be open")

        except FileNotFoundError:
            my_label.config(text="File Not Found !!")

    # clear the Tree view
    clear_tree()

    # set up new Tree view
    my_tree["column"] = list(df.columns)
    my_tree["show"] = "headings"

    # loop thru column list

    for column in my_tree["column"]:
        my_tree.heading(column,text=column)

    df_rows = df.to_numpy().tolist()

    for row in df_rows:
        my_tree.insert("","end",values=row)

    my_tree.grid(row=0,column=0,pady=10)
    btn_close.grid(row=1, column=0)



def clear_tree():
    my_tree.delete(*my_tree.get_children())

label = Label(frame1)
# label_two = Label(frame1)

photo = ImageTk.PhotoImage(image)
# photo_two = ImageTk.PhotoImage(image_two)

label['image'] = photo
# label_two['image'] = photo_two

label.grid(row=0,column=0,pady=5)
# label_two.grid(row=0,column=1,pady=5)

#
# btn_face_mask_detection = Button(frame2,text="Face MasK Detection")
# btn_face_mask_detection.grid(row=0,column=0,padx=20,pady=20)


btn_face_mask_recognition = Button(frame2,text="Face Mask Recognition",command=face_mask_detetction_fun)
btn_face_mask_recognition.grid(row=0,column=0,padx=20,pady=20)

btn_open = Button(frame2,text="Open",command=file_open)
btn_open.grid(row=0,column=1,padx=20,pady=20)

btn_close = Button(frame3,text="Close",command=clear_tree)


my_tree = ttk.Treeview(frame3)

my_label = Label(root,text='')
my_label.pack(pady=20)

root.geometry("1500x800")

root.mainloop()