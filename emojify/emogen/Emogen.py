#(0=Angry, 1=Fear, 2=Happy, 3=Sad, 4=Surprise, 5=Neutral)
import numpy as np
import cv2
import keras.models
import keras
import glob
from numpy import array
import imutils
import sys
import cv2

# load json and create model
json_file = open('../training/modelgenerated.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = keras.models.model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("../training/modelgenerated.h5")
print("Loaded model from disk")

# using the model to generated emojis on live faces

loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#using haarcascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)
roi_gray=[] 
emoji=[]   
pop=0      
pr=0       
while 1:     
    ret, img = cap.read()   
    
    #Creating GreyScale Image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    center=0
    for (x,y,w,h) in faces:
        if(w>h):
            big=w
        else:
            big=h
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        if(pop==1):
            emoji=cv2.imread("TempImg.jpg")
            emoji= cv2.resize(emoji,(w,h))
            #print(img[y:y+h,x:x+w])
            img[y:y+h,x:x+w]= emoji  
            #print(emoji)
    k = cv2.waitKey(30) & 0xff
    if(k== 13):
        #Detecting Emotion on pressing enter key

        temp = cv2.resize(roi_gray,(48,48))
        cv2.imwrite("TempImg.jpg",roi_color)
        cv2.imwrite("facehai.jpg",temp)
        temp= cv2.imread("facehai.jpg")
        temp= temp/255
        pr=loaded_model.predict_classes(temp.reshape(1,48,48,3))
        #setting flag pop to 1
        pop=1
    
    img = imutils.resize(img, width=1280)   
    cv2.imshow('img',img)
    
    #Stopping Excecution on escape key
    if k == 27:
        break
cap.release()    
cv2.destroyAllWindows()
