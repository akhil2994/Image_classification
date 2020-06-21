import os
import numpy as np
import keras
import PIL
import tensorflow
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from tensorflow.keras.models import model_from_json

#load model
json_file = open("models/model_3.json", 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("models/model_3.h5")


#load and prepare the image
def load_image(filename):
    img = load_img(filename, target_size=(32, 32))
    img = img_to_array(img)
    img = img.reshape(1, 32, 32, 3)     #reshape into a single sample with 3 channels
    img = img.astype('float32')
    img = img / 255.0
    return img


#define labels
def get_label(argument):
    labels = {0:'Airplane', 1:'Automobile', 2:'Bird', 3:'Cat', 4:'Deer', 5:'Dog',
              6:'Frog', 7:'Horse', 8:'Ship', 9:'Truck'}
    return(labels.get(argument, "Invalid !!"))


img = load_image('test/car.jpg')
result = loaded_model.predict_classes(img)
value = get_label(result[0])
print(value)