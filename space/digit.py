#!/usr/bin/env python
import numpy as np
import cv2
# import scipy.ndimage as nd

readimage = cv2.imread('digit.png')
ret, thres = cv2.threshold(readimage, 127, 255, cv2.THRESH_BINARY_INV)
cut = np.array(thres[210:260,120:170])
# cuts = nd.interpolation.zoom(cut,2/5,order=1)
print(cut.shape)
cv2.imshow('myimage', cut)
cv2.imwrite('myimage.png', cut)
cv2.waitKey(0) & 0xFF
	    
cv2.destroyAllWindows()
