#!/usr/bin/env python
import scipy.io as sio
import numpy as np
import cv2
import glob
from constants import *
import time
from scipy.stats import logistic
from scipy.special import expit
from oct2py import octave as oc
import os

camera = cv2.VideoCapture(0)

def getImage():
    ret, image = camera.read()
    return image


def color(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image

def resize(image):
    image = cv2.resize(image, outputSize)
    return image

def predict(Theta1, Theta2, X):
    # h1 = expit()
    # X = np.transpose(X)
    X = np.insert(X, 0, 1)
    h1 = expit(np.dot(X, np.transpose(Theta1)))
    h1 = np.insert(h1, 0, 1)
    h2 = expit(np.dot(h1, np.transpose(Theta2)))
    return h2.argmax() + 1


def getFace(image):
    cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    minisize = (image.shape[1], image.shape[0])
    miniframe = cv2.resize(image, minisize)
    face = cascade.detectMultiScale(miniframe)
    if face is not None and len(face)>0:
        face = face[0]
        x, y, w, h = [v for v in face]
        face = image[y:y + h, x:x + w]
        if(h > ymin and w > xmin):
            cv2.rectangle(image,(x,y),(x+w,y+w),(0,0,0))
            return face

def createFolder(folderName):
    if not os.path.exists(folderName):
        os.mkdir(folderName)

def test():
    createFolder("newPic")
    count=0
    while True:
        image=getImage()
        count+=1
        cv2.imshow('test',image)
        cv2.imwrite("newPic/"+str(count)+".jpg",image)
        time.sleep(1)
    camera.release()


def main():
    count = 0
    while True:
        image = getImage()
        face = getFace(image)
        if face is not None:
            face=resize(face)
            image[0:face.shape[0],0:face.shape[1]]=face
            face=color(face)
            cv2.imwrite('temp.jpg',face)
            count += 1
            print(str(count) + " : " +
                  numToFace[int(oc.predictPy('temp.jpg'))])
        cv2.imshow('image', image)
        k = cv2.waitKey(1)
        if k == ord('q'):
            break
        # time.sleep(1)
        # break
    cv2.destroyAllWindows()
    camera.release()


if __name__ == "__main__":
    # test()
    main()
