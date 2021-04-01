import serial

arduino_port = "/dev/ttyUSB0"  # serial port of Arduino
baud = 9600  # arduino uno runs at 9600 baud
fileName="analog-data-10.csv" #name of the CSV file generated

ser = serial.Serial(arduino_port, baud)
print("Connected to Arduino port:" + arduino_port)
file = open(fileName, "a")
print("Created file")

#display the data to the terminal
while True:
    ser_bytes = ser.readline()
    decoded_bytes = float(ser_bytes[0:len(ser_bytes)-2])
    data = str(decoded_bytes)
    print(data)
    file = open(fileName, "a") #append the data to the file
    file.write(data + "\n") #write data with a newline



#add the data to the file
# print("\\n")

#close out the file
file.close()