from collections import deque
from random import randint

import cv2
import imutils
import numpy as np



video = cv2.VideoCapture("http://192.168.0.11:4747/mjpegfeed")
HISTLEN=10
history = deque(maxlen=HISTLEN)
diffA=0.7
diffB=1.3

ok, image=video.read()
lower=[]
upper=[]
colors=[]
while True:
    bbox = cv2.selectROI('Tracker', image)
    (x, y, w, h) = bbox
    roi = image[y:y + h, x:x + w]
    if roi.size > 0:
        hsvRoi=(cv2.cvtColor(roi, cv2.COLOR_BGR2HSV))
        lower.append( np.array([ np.percentile( hsvRoi[:, :, 0],1 ) , np.percentile(hsvRoi[:, :, 1], 1), np.percentile(hsvRoi[:, :, 2], 1)])*diffA)
        colors.append((randint(0, 255), randint(0, 255), randint(0, 255)))
        upper.append(np.array([np.percentile(hsvRoi[:, :, 0], 99), np.percentile(hsvRoi[:, :, 1], 99),
                               np.percentile(hsvRoi[:, :, 2], 99)] )* diffB)

    print("Press q to quit selecting boxes and start tracking")
    print("Press any other key to select next object")
    k = cv2.waitKey(0) & 0xFF
    if k == 113:  # q is pressed
        break


while True:
    ok, frame = video.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask=[]


    for i, lowerI in enumerate(lower):
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.inRange(hsv, lowerI, upper[i])
        mask = cv2.erode(mask, None, iterations=3)
        mask = cv2.dilate(mask, None, iterations=3)


        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if imutils.is_cv2() else cnts[1]
        center = None
        if len(cnts) > 0:
            c = max(cnts, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            M = cv2.moments(c)
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            print(center)

            # only proceed if the radius meets a minimum size
            if radius > 10:
                cv2.circle(frame, (int(x), int(y)), int(radius),
                           colors[i], 2)
                cv2.circle(frame, center, 5, (0, 0, 255), -1)



    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break


    cv2.waitKey(1)

