from random import randint

import cv2
import sys

from tracking import createTrackerByName

video = cv2.VideoCapture("http://192.168.0.11:8080/mjpeg")

if not video.isOpened():
    print("Nie można otworzyć video")
    sys.exit()

bboxes = []
colors = []
ok, image = video.read()
# Wczytywanie obiektów do śledzenia
while True:
    bbox = cv2.selectROI('Tracker', image)
    (x, y, w, h) = bbox
    roi = image[y:y + h, x:x + w]
    if roi.size > 0:
        bboxes.append(bbox)
        colors.append((randint(0, 255), randint(0, 255), randint(0, 255)))
    print("Press q to quit selecting boxes and start tracking")
    print("Press any other key to select next object")
    k = cv2.waitKey(0) & 0xFF
    if k == 113:
        break

multiTracker = cv2.MultiTracker_create()
# Initialize MultiTracker
for bbox in bboxes:
    multiTracker.add(createTrackerByName("CSRT"), image, bbox)


while True:
    ok, frame = video.read()
    if not ok:
        print("Brak obrazu")
        continue

    timer = cv2.getTickCount()
    ok, boxes = multiTracker.update(frame)
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

    # Narysuj wynik
    if ok:
        for i, newbox in enumerate(boxes):
            p1 = (int(newbox[0]), int(newbox[1]))
            p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
            cv2.rectangle(frame, p1, p2, colors[i], 2, 1)
    else:
        cv2.putText(frame, "Błąd śledzenia", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
    cv2.putText(frame, "FPS : " + str(int(fps)), (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1)
    cv2.imshow("Tracking", frame)

    k = cv2.waitKey(1) & 0xff
    if k == 23: break