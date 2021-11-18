import cv2
from cvzone.HandTrackingModule import HandDetector
import cvzone
import numpy as np

captureDevice = cv2.VideoCapture(0, cv2.CAP_DSHOW)
captureDevice.set(3, 1280)
captureDevice.set(4, 720)
detector = HandDetector(detectionCon=1, maxHands=2)
color = (0, 0, 0)
cx, cy, width, height = 100, 100, 200, 200


class DragAndDrop():
    def __init__(self, positionCenter, size=[200, 200]):
        self.posCenter = positionCenter
        self.size = size

    def update(self, cursor):
        cx, cy = self.posCenter
        width, height = self.size

        if cx - width // 2 < cursor[0] < cx + width // 2 and cy - height // 2 < cursor[1] < cy + height // 2:
            self.posCenter = cursor


rectangleList = []
for x in range(3):
    rectangleList.append(DragAndDrop([x * 250 + 300, 250]))

while True:
    success, img = captureDevice.read()
    img = cv2.flip(img, 1)
    img = detector.findHands(img)
    lmList, _ = detector.findPosition(img)

    if lmList:
        l, _, _ = detector.findDistance(8, 12, img, draw=False)
        if l < 60:
            cursor = lmList[8]
            for rectangle in rectangleList:
                rectangle.update(cursor)

    imgNew = np.zeros_like(img, np.uint8)
    for rectangle in rectangleList:
        cx, cy = rectangle.posCenter
        width, height = rectangle.size
        cv2.rectangle(imgNew, (cx - width // 2, cy - height // 2), (cx + width // 2, cy + height // 2), color,
                      cv2.FILLED)
        cvzone.cornerRect(imgNew, (cx - width // 2, cy - height // 2, width, height), 100, rt=0)
    out = img.copy()
    alpha = 0.1
    mask = imgNew.astype(bool)
    out[mask] = cv2.addWeighted(img, alpha, imgNew, 1 - alpha, 0)[mask]

    cv2.imshow("Image", out)
    cv2.waitKey(1)
