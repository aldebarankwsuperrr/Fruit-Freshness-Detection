import cv2
import torch
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from flask import Flask, request, jsonify, render_template
import urllib
app =Flask(__name__)


classification_model = tf.keras.models.load_model(
    './model.h5',
    custom_objects={'KerasLayer':hub.KerasLayer}
)
object_detection_model= torch.hub.load('ultralytics/yolov5', 'custom', './model_v3_1_0.pt')

label_classes = ['freshapples','freshbanana','freshoranges','rottenapples','rottenbanana','rottenoranges']
label_object_detection = ['apple', 'orange', 'banana']


def url_to_image(url):
	# download the image, convert it to a NumPy array, and then read
	# it into OpenCV format
	resp = urllib.request.urlopen(url)
	image = np.asarray(bytearray(resp.read()), dtype="uint8")
	image = cv2.imdecode(image, cv2.IMREAD_COLOR)
	# return the image
	return image

@app. route("/predict", methods = ["POST"])
def predict():
  data = request.json
  url = data['url']
  image = url_to_image(url)
  result = object_detection_model(image)
  fruit_name  = []
  freshness = []
  for detect in result.xyxy[0]:
    xB = int(detect[2])
    xA = int(detect[0])
    yB = int(detect[3])
    yA = int(detect[1])
    label_index = int(detect[5])
    label = label_object_detection[label_index]   
    fruit_name.append(label)
     
    load = image[yA:yB, xA:xB]
    
    load = load/255.0
    load = cv2.resize(load, (224,224))
    z = tf.keras.utils.img_to_array(load)
    z = np.expand_dims(z, axis=0)
    images = np.vstack([z])
    classes = classification_model.predict(images)
    index = np.argmax(classes) 
    freshness_level = label_classes[index]
    freshness.append(freshness_level)
  return jsonify({"name": fruit_name,
                    "freshness": freshness})   
  # return jsonify({"tes": "halo"})
    
if __name__ == "__main__":
  app.run(debug=True)
