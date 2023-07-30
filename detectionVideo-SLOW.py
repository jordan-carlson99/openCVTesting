import cv2 as cv
import numpy as np
import time


def rescaleFrame(frame, scale=0.75):
    height = int(frame.shape[0] * scale)
    width = int(frame.shape[1] * scale)

    dimensions = (height, width)

    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)


# initialize yolo weights ---

# get a list of the classes for detection
classes = open("coco.names").read().strip().split("\n")

# sets initial state for random number generation
np.random.seed(42)
# create list of random hex values to be used for colors
colors = np.random.randint(0, 255, size=(len(classes), 3), dtype="uint8")

# load neural network into cv object - uses darknet open source framework
net = cv.dnn.readNetFromDarknet("yolov3-tiny.cfg", "yolov3-tiny.weights")
# this sets what hardware would be running the computation, opencv is generally good for cpus
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)

# this gets the names for the weights of the output layers so we can read them later
ln = net.getLayerNames()
ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

# get the video from device 0 (webcam)
live = cv.VideoCapture(0)

while True:
    # capture frame as an image
    isTrue, img = live.read()

    # blob = binary large object - turn the image into a multi-dimensional array cointaining
    # the important parts of image so the network can reduce and apply models to the data
    blob = cv.dnn.blobFromImage(img, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    # 0 retrieves first image, 0 accesses the grayscale channel, : means include all height and
    # width values. set the image to grayscale, r returns 2d array of the image
    r = blob[0, 0, :, :]
    # print(r)

    # print(blob.shape)
    # text = f'Blob shape={blob.shape}'
    # cv.displayOverlay('blob', text)

    # put blob into neural network, "load" it
    net.setInput(blob)
    # t0 = time.time()
    # put the blob through the layers and capture the output specified, "shoot" it
    outputs = net.forward(ln)
    # print(outputs)
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            print(confidence)
    # t = time.time()
    # print('time = ', t-t0)

    # for out in outputs:
    #     print(out.shape)

    cv.imshow("video", r)

    if cv.waitKey(20) & 0xFF == ord("q"):
        break
live.release()
cv.destroyAllWindows()
