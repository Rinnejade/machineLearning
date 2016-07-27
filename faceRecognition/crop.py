#!/usr/bin/env python3
'''  
facechop.py

-Takes an image and detects a face in it.  
-For each face, an image file is generated
    -the images are strictly of the faces
'''

import cv2
import glob
import os
from constants import *

def facechop(image, code, outFolder):
    imageName = image.split("/")[1]
    facedata = "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(facedata)

    img = cv2.imread(image)

    minisize = (img.shape[1], img.shape[0])
    miniframe = cv2.resize(img, minisize)

    faces = cascade.detectMultiScale(miniframe)

    for f in faces:
        x, y, w, h = [v for v in f]
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255))
        sub_face = colorResize(img[y:y + h, x:x + w])
        if(h > ymin and w > xmin):
            face_file_name = outFolder + code + "_" + str(y) + "_" + imageName
            cv2.imwrite(face_file_name, sub_face)


def facecrop(imagePattern, code, outFolder):
    imgList = glob.glob(imagePattern)
    if len(imgList) <= 0:
        print('No Images Found')
        return

    for img in imgList:
        facechop(img, code, outFolder)


def init():
    if not os.path.exists("face"):
        os.mkdir("face")
    if not os.path.exists("test"):
        os.mkdir("test")


def colorResize(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, outputSize)
    return image

def test():
    facecrop("test/*.jpg","1","test/")

if __name__ == '__main__':
    init()
    # facecrop("Nisham_Mohammed/*.jpg", "1", "face/")
    # facecrop("Kevin_Joseph/*.jpg", "2", "face/")
    facecrop("newPic/*.jpg", "1", "face/")
    test()
