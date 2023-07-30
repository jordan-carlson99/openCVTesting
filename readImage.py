import cv2 as cv
import numpy as np
import time

# create img object to access open cv methods and properties
img = cv.imread("./image/person.JPG")

# image recognition ---

# net = cv.dnn.readNetFromDarknet('yolov3.cfg', 'yolov3.weights')
# net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)

# ln = net.getLayerNames()
# print(len(ln), ln)

# blob = cv.dnn.blobFromImage(img, 1/255.0, (416, 416), swapRB=True, crop=False)
# r = blob[0, 0, :, :]

# #    the image to transform
# #    the scale factor (1/255 to scale the pixel values to [0..1])
# #    the size, here a 416x416 square image
# #    the mean value (default=0)
# #    the option swapBR=True (since OpenCV uses BGR)

# cv.imshow('blob', r)
# text = f'Blob shape={blob.shape}'
# cv.displayOverlay('blob', text)
# cv.waitKey(1)

# net.setInput(blob)
# t0 = time.time()
# outputs = net.forward(ln)
# t = time.time()

# cv.displayOverlay('window', f'forward propagation time={t-t0}')
# cv.imshow('window',  img)
# cv.waitKey(0)
# cv.destroyAllWindows()


# function for rescaling frame (default is 0.75)


# creates new window to display img, first arg= name of window, second is object to display
scale = 0.1
img = cv.resize(img, (0, 0), fx=scale, fy=scale)
cv.imshow("person", img)


# listen for key to close
cv.waitKey(0)
