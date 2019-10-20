#example input = python final.py --conf conf.json

import numpy as np
import cv2
import RPi.GPIO as GPIO
import time
from tempFiles.tempimage import TempImage
from picamera.array import PiRGBArray
from picamera import PiCamera
import argparse
import warnings
import datetime
import dropbox
import imutils
import json
import time
import cv2

#Constructs a parser for the arguments on compiling
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--conf", required=True)
args = vars(ap.parse_args())
warnings.filterwarnings("ignore")
conf = json.load(open(args["conf"]))
client = None

#Check to see if system is enabled for dropbox or if wanted to test
if conf["dropbox_enabled"]:
	#Connects to lined dropbox app account
	client = dropbox.Dropbox(conf["dropbox_access_token"])
	print("Linked to dropbox app account")
	
print("initializing setup")
print("press escape in video terminal to exit program")
time.sleep(conf["system_setup"])
avg = None
lastImageSent = datetime.datetime.now()
faceDetected = 0
# Obtain the Haar cascade data from OpenCV
faceCascade = cv2.CascadeClassifier('/home/pi/opencv/data/haarcascades/haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)
CAMLED = 32
# Sets width of camera stream
cap.set(3,640)
# Sets height of camera stream
cap.set(4,480)
#Initilaizes the LEDs
GPIO.setmode(GPIO.BCM)
GPIO.setup(17, GPIO.OUT)
GPIO.setup(18, GPIO.OUT)
GPIO.setup(22, GPIO.OUT)
#Turns off the LEDS before the system starts
GPIO.output(17,False)
GPIO.output(18,False)
GPIO.output(22,False)


while True:
     #Captures the video from the camera.
    ret, img = cap.read()
    imgTimeStamp = datetime.datetime.now()
    #Varibale for telling system iof it detects face
    detected = "no"
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #Calls the cascade function and passed the parameters
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,     
        minSize=(20, 20)
    )
    #LED transistion
    GPIO.output(22, False)
    GPIO.output(18,False)
    GPIO.output(17,True)
    #Saves imaghe time stamp of when face was detected
    ts = imgTimeStamp.strftime("%A %d %B %Y %I:%M:%S%p")
    #Draws rectangle around detected face
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        #LED transsition
        GPIO.output(17,False)
        GPIO.output(18,True)
        #Pauses to verify
        time.sleep(1)
        #Changes detected vaiable to yes
        detected = "yes"
        
         #Begins the data upload process
        if detected == "yes":
	    #Makes sure the upload delay has occurred before uploading again
            if (imgTimeStamp - lastImageSent).seconds >= conf["upload_delay"]:
		
                faceDetected += 1
		#Cheks to see if the dropbox account is linked then uploades
                if conf["dropbox_enabled"]:
			#Writes the captured images to a temp file
                        t = TempImage()
                        cv2.imwrite(t.path, img)
                        #LED transition
                        GPIO.output(18, False)
                        GPIO.output(22, True)
			#Uploads the image to the App dropbox acount
			#Saves the image as the date and imgTimeStamp
                        print("Uploaded file: {}".format(ts))
                        path = "/{base_path}/{imgTimeStamp}.jpg".format(
			    base_path=conf["dropbox_online_filepath"], imgTimeStamp=ts)
                        client.files_upload(open(t.path, "rb").read(), path)
                        #Clears the temp image file
                        t.cleanup()
			#Updates the last uploaded time to the imgTimeStamp
			#resets the detected counter
                        lastImageSent = imgTimeStamp
                        faceDetected = 0

                else:
                    faceDetected = 0
    cv2.imshow('video',img)

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        #When Escape is hit the program ends
        break

cap.release()
cv2.destroyAllWindows()