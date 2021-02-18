import cv2

#from model import FacialExpressionModel
from model import Mask_NoMask
import numpy as np

#load the 'haarcascade_frontalface' OpenCV model classifier
facec = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#Path were model is stored
#modify the path where you download the model
model_path="best trained models"

#Create an object of class Mask_NoMask.py
model = Mask_NoMask()
font = cv2.FONT_HERSHEY_SIMPLEX

'''
    # a method to capture frames from a video feed
    # in a frame capture all the faces
    # crop each face and format it so that it can be used for model prediction
    # send the cropped face for model to predict
    # use model prediction to sketch box and label the face in live video feed
'''

class VideoCamera(object):
    def __init__(self):
        #self.video = cv2.VideoCapture('videos/facial_exp.mkv')
        self.video = cv2.VideoCapture(0)
    def __del__(self):
        self.video.release()

    def get_frame(self):
        _, fr = self.video.read()
        if fr is not None:
            gray_fr = cv2.cvtColor(fr, cv2.COLOR_RGB2BGR)#Changed from RGB2BGR to RGB2RGB
            faces = facec.detectMultiScale(gray_fr, 1.3, 5)
        else:
            print("Empty Frame")
        
        #Loop through all the faces found in a video frame
        #format the image to be used for model prediction (resize to 128X128X3, reshape and normalize by /255)
        for (x, y, w, h) in faces:
            fc = gray_fr[y:y + h, x:x + w]

            #resize the image to 128X128 (for prediction, model accpets the image size in 128X128)
            roi = cv2.resize(fc, (128, 128))
            roi = np.reshape(roi, [1, 128, 128, 3])/255.0
            
            #pass the image to model for prediction
            pred = model.predict_mask(roi)
            
            #Determine the color of box and text to be shown around the face in live video feed
            if pred=='Mask':
                cv2.putText(fr, pred, (x, y), font, 1, (3,252,111), 2)
                cv2.rectangle(fr,(x,y),(x+w,y+h),(3,252,111),2)
            else:
                cv2.putText(fr, pred, (x, y), font, 1, (255, 0, 47), 2)
                cv2.rectangle(fr,(x,y),(x+w,y+h),(255, 0, 47),2)
        
        _, jpeg = cv2.imencode('.jpg', fr)
        return jpeg.tobytes()
