## from firebase import firebase
import cv2
import numpy as np
import os
import playsound
import threading
import time
import pygame
import smtplib


recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);

font = cv2.FONT_HERSHEY_SIMPLEX

id = 0
id2 = 0

names = ['Unknown', 'Human']

cam = cv2.VideoCapture(0)
cam.set(3, 450)
cam.set(4, 380)

minW = 0.05*cam.get(3)
minH = 0.05*cam.get(4)


while True:

    ret, img =cam.read()

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor = 1.2,
        minNeighbors = 5,
        minSize = (int(minW), int(minH)),
       )

    for(x,y,w,h) in faces:

        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)

        id, confidence = recognizer.predict(gray[y:y+h,x:x+w])
        print(confidence)
        if (confidence < 120):
            # if (id==1 or id==2) and (confidence<110):
                # pygame.init()
                # time.sleep(1)
                # playsound.playsound("alarm.wav", block = "False")
            id = names[id]
            confidence = "  {0}%".format(round(confidence))

        else:
            id = "unknown"
            confidence = "  {0}%".format(round(confidence))

        cv2.putText(img, str(id), (x+5,y-5), font, 1, (255,255,255), 2)
        cv2.putText(img, str(confidence), (x+5,y+h-5), font, 1, (255,255,0), 1)



    cv2.imshow('camera',img)
    exitButton = cv2.waitKey(10) & 0xff
    if exitButton == 27:
        break

print("\n Stopping the programm ...")
cam.release()
cv2.destroyAllWindows()

