import cv2 as cv


# 0 means video device
capture = cv.VideoCapture(0)


# NOTE: only works on live video
def changeRes(width, height):
    # 3 and 4 reference the properties of capture class
    capture.set(cv.CAP_PROP_FRAME_WIDTH, width)
    capture.set(4, height)
    capture.set(5, 5)
    return capture


capture = changeRes(255, 255)

# loop because open cv reasds video frame by frame and processes each as an image
while True:
    # istrue and frame are both the frame of the video
    isTrue, frame = capture.read()

    # video is shown using image showing method of the captured frame
    cv.imshow("video", frame)

    # if letter 'D' is pressed break loop
    if cv.waitKey(20) & 0xFF == ord("q"):
        break

# release the capture device and close
capture.release()
cv.destroyAllWindows()
