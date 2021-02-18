#from tensorflow.keras.models import model_from_json
from tensorflow.python.keras.backend import set_session
import numpy as np
from keras.preprocessing import image
from tensorflow import keras

import tensorflow as tf

config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.15
session = tf.compat.v1.Session(config=config)
set_session(session)

class Mask_NoMask():

    def __init__(self):
            #change the path where you download the model
            model_path="best trained models"
            print(model_path)
            #load the model
            self.loaded_model = keras.models.load_model(model_path)


    def predict_mask(self, img):
        global session
        set_session(session)
        self.preds = round(self.loaded_model.predict(img)[0][0])
        if self.preds==0:
            return "Mask"
        else:
            return "No-Mask"