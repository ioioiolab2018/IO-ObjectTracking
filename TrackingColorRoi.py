from collections import deque
from random import randint

import cv2
import imutils
import numpy as np

video = cv2.VideoCapture("http://192.168.0.11:8080/mjpeg")

diffA = 0.7
diffB = 1.3

lower = []
upper = []
colors = []



ok, image = video.read()
# Wczytywanie obiektów do śledzenia
while True:
    bbox = cv2.selectROI('Tracker', image)
    (x, y, w, h) = bbox
    roi = image[y:y + h, x:x + w]
    # Wyznaczenie koloru śledzonego obiektu
    if roi.size > 0:
        hsvRoi = (cv2.cvtColor(roi, cv2.COLOR_BGR2HSV))
        lower.append(np.array([np.percentile(hsvRoi[:, :, 0], 1), np.percentile(hsvRoi[:, :, 1], 1),
                               np.percentile(hsvRoi[:, :, 2], 1)]) * diffA)
        upper.append(np.array([np.percentile(hsvRoi[:, :, 0], 99), np.percentile(hsvRoi[:, :, 1], 99),
                               np.percentile(hsvRoi[:, :, 2], 99)]) * diffB)
        colors.append((randint(0, 255), randint(0, 255), randint(0, 255)))

    print("Press q to quit selecting boxes and start tracking")
    print("Press any other key to select next object")
    k = cv2.waitKey(0) & 0xFF
    if k == 113:
        break

while True:
    timer = cv2.getTickCount()
    ok, frame = video.read()
    fps=0
    if ok:
        blurFrame = cv2.GaussianBlur(frame, (15, 15), 0)
        hsv = cv2.cvtColor(blurFrame, cv2.COLOR_BGR2HSV)

        for i, lowerI in enumerate(lower):
            mask = cv2.inRange(hsv, lowerI, upper[i])
            mask = cv2.erode(mask, None, iterations=3)
            mask = cv2.dilate(mask, None, iterations=3)
            cv2.imshow("MASK: "+str(i),mask)
            contours = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
            contours = contours[0] if imutils.is_cv2() else contours[1]
            center = None
            if len(contours) > 0:
                maxContour = max(contours, key=cv2.contourArea)
                rect = cv2.minAreaRect(maxContour)
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                cv2.drawContours(frame, [box], 0, colors[i], 2)
                M = cv2.moments(maxContour)
                center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                cv2.circle(frame, center, 5, (0, 0, 255), -1)

        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
        cv2.putText(frame, "FPS : " + str(int(fps)), (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1);
        cv2.imshow("Frame", frame)
    else:
        print("Brak obrazu")
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

    cv2.waitKey(1)
