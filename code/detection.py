import cv2
import numpy as np
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

def face_mask_detetction_fun():
    net = cv2.dnn.readNet('yolov3_training_final.weights', 'yolov3_testing.cfg')

    classes = []
    font = cv2.FONT_HERSHEY_DUPLEX
    with open('classes.txt', 'r') as f:
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

        if len(indexes) > 0:
            for i in indexes.flatten():
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                confidence = str(round(confidences[i], 2) * 100)
                if label == "not wearing mask":
                    color = (0, 0, 255)
                    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
                        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
                        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)

                        matchIndex = np.argmin(faceDis)

                        if matches[matchIndex]:
                            name = classNames[matchIndex].upper()
                            # print(name)

                            markAttendence(name)
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

def append_data(lst):
    df = pd.DataFrame(lst)
    df.to_csv('Attendence.csv', mode='a', index=False,header=False)
    remove_duplicates()

def remove_duplicates():
    df = pd.read_csv('Attendence.csv')
    df.drop_duplicates(inplace=True)
    df.to_csv('Attendence.csv', mode='w',index=False)

def markAttendence(name):
    mask = "Not Wearing"
    temp = 37.25
    get_time = datetime.now().time().strftime("%H%M")  # now
    get_time = datetime.strptime(get_time, "%H%M").time()
    lst = {'Name': [name], 'Mask Status': [mask], 'Temperature': [temp], 'Date': [datetime.now().date()],
           'Time': [get_time]}
    append_data(lst)


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
frame2.place(x=10,y=440)
frame3.pack(side=BOTTOM)
frame3.place(x=340,y=400)


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

    my_tree.grid(row=0,column=0)
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