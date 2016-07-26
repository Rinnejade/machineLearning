#!/usr/bin/env python
import cv2
from PIL import Image
import glob
import os


def detectFace(image, faceCascade, returnImage=False):
    min_size = (20, 20)
    image_scale = 1
    haar_scale = 1.1
    min_neighbors = 3
    haar_flags = 0

    cv2.equalizeHist(image, image)

    faces = cv2.HaarDetectObjects(image, faceCascade, cv.CreateMemStorage(
        0), haar_scale, min_neighbors, haar_flags, min_size)

    if(faces and returnImage):
        for((x, y, w, h), n) in faces:
            pt1 = (int(x), int(y))
            pt2 = (int(x + w), int(y + h))
            cv2.Rectangle(image, pt1, pt2, cv.RGB(255, 0, 0), 5, 8, 0)

    if returnImage:
        return image
    else:
        return faces


def pil2cvgray(pil_im):
    pil_im = pil_im.convcv_im = cv.CreateImageHeader(
        pil_im.size, cv.IPL_DEPTH_8U, 1)
    cv_im = cv2.CreateImageHeader(pil_im.size, cv.IPL_DEPTH_8U, 1)
    cv2.SetData(cv_im, pil_im.tostring(), pil_im.size[0])
    return cv_im


def cv2pil(cv_im):
    return Image.fromstring("L", cv2.GetSize(cv_im), cv_im.tostring())


def imgCrop(image, cropBox, boxScale=1):
    xDelta = max(cropBox[2] * (boxScale - 1), 0)
    yDelta = max(cropBox[3] * (boxScale - 1), 0)
    PIL_box = [cropBox[0] - xDelta, cropBox[1] - yDelta, cropBox[0] +
               cropBox[2] + xDelta, cropBox[1] + cropBox[3] + yDelta]
    return image.crop(PIL_box)


def faceCrop(imagePattern, boxScale=1):
    faceCascade = cv2.Load('haarcascade_frontalface_alt.xml')
    imgList = glob.glob(imagePattern)
    if len(imgList) <= 0:
        print('No Images Found')
        return

    for img in imgList:
        pil_im = Image.open(img)
        cv_im = pil2cvGrey(pil_im)
        faces = DetectFace(cv_im, faceCascade)
        if faces:
            n = 1
            for face in faces:
                croppedImage = imgCrop(pil_im, face[0], boxScale=boxScale)
                fname, ext = os.path.splitext(img)
                croppedImage.save(fname + '_crop' + str(n) + ext)
                n += 1
        else:
            print('No faces found:', img)


def test(imageFilePath):
    pil_im = Image.open(imageFilePath)
    cv_im = pil2cvGrey(pil_im)
    faceCascade = cv2.Load('haarcascade_frontalface_alt.xml')
    face_im = DetectFace(cv_im, faceCascade, returnImage=True)
    img = cv2pil(face_im)
    img.show()
    img.save('test.png')

faceCrop('images/*.jpg', boxScale=1)
