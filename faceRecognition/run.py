#!/usr/bin/env python
import scipy.io as sio
import numpy as np
import cv2
from constants import *
import time
from scipy.stats import logistic
from scipy.special import expit

camera = cv2.VideoCapture(0)


def getImage():
    ret, image = camera.read()
    return image


def colorResize(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
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


def getFaces(image):
    cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    minisize = (image.shape[1], image.shape[0])
    miniframe = cv2.resize(image, minisize)
    faces = cascade.detectMultiScale(miniframe)
    faceList = []
    for f in faces:
        x, y, w, h = [v for v in f]
        face = colorResize(image[y:y + h, x:x + w])
        if(h > ymin and w > xmin):
            faceList.append(face)
    return faceList


def main():
    count = 0
    Theta1 = np.loadtxt('Theta1.mat')
    Theta2 = np.loadtxt('Theta2.mat')
    while True:
        image = getImage()
        faces = getFaces(image)
        for face in faces:
            cv2.imshow('camera', face)
            face = np.ravel(face)
            count += 1
            print(str(count) + " : " +
                  numToFace[predict(Theta1, Theta2, face)])
        k = cv2.waitKey(1)
        if k == ord('q'):
            break
        # time.sleep(1)
        # break
    cv2.destroyAllWindows()
    camera.release()


if __name__ == "__main__":
    main()
