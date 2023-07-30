import cv2 as cv
import torch

model = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)


def plot_boxes(results, frame):
    labels, cords = results

    for cord in cords:
        x, y, w, h, conf = cord

        x1, y1 = int(x * frame.shape[1]), int(y * frame.shape[0])
        x2, y2 = int((x + w) * frame.shape[1]), int((y + h) * frame.shape[0])

        cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv.putText(
            frame,
            f"{int(conf*100)}% {labels}",
            (x1, y1 - 10),
            cv.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2,
        )

    return frame


capture = cv.VideoCapture(0)

while True:
    capture.set(cv.CAP_PROP_FPS, 10)
    isTrue, img = capture.read()
    scale = 0.2
    img = cv.resize(img, (0, 0), fx=scale, fy=scale)
    results = model(img)
    results = (results.xyxyn[0][:, -1].numpy(), results.xyxyn[0][:, :-1].numpy())

    img = plot_boxes(results=results, frame=img)
    # print(results)
    cv.imshow("rendered image", img)
    if cv.waitKey(20) & 0xFF == ord("q"):
        break

capture.release()
cv.destroyAllWindows()

# results.print()
# cv.imshow("image", img)
