from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import cv2
import os, os.path

imageFolder = "images/"

validImages = [".jpg",".jpeg"]

face_cascade = cv2.CascadeClassifier("./cascade/haarcascade_frontalface_default.xml")

model = load_model('keras_model.h5', compile=False)

def getImages():

    images = []

    for f in os.listdir(imageFolder):
        ext = os.path.splitext(f)[1]
        if ext.lower() not in validImages:
            continue
        images.append(f)

    return images

def predict(imagePath):
    image = Image.open(imagePath)
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
    data[0] = normalized_image_array

    prediction = model.predict(data)

    classes = prediction.argmax(axis=-1)

    return classes

def identifyFaces(image):
    imgCv2 = cv2.imread(imageFolder + image)
    gray = cv2.cvtColor(imgCv2, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    detectedClasses = []

    for (x, y, w, h) in faces:
        img = Image.fromarray(imgCv2)
        imgFace = img.crop((x, y, (x + w), (y + h)))
        size = (224, 224)
        imgFace = ImageOps.fit(imgFace, size, Image.ANTIALIAS)
        image_array = np.asarray(imgFace)
        normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        data[0] = normalized_image_array
        prediction = model.predict(data)

        classes = prediction.argmax(axis=-1)

        detectedClasses.append(classes)

    return detectedClasses

for image in getImages():

    classes = identifyFaces(image)
    classesString = ','.join(map(str, classes)) 

    print("For image " + image + ": " + classesString)
















