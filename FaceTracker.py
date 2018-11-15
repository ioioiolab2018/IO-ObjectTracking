import math

import cv2

from imutils.video import WebcamVideoStream

csrtTracker = cv2.TrackerCSRT_create()
CHECK = 20
video = WebcamVideoStream(src="http://192.168.1.11:8080/mjpeg").start()
face_cascade = cv2.CascadeClassifier()
face_cascade.load("haarcascade_frontalface_default.xml")
lastpos = [0, 0]
lastface = []


def findface(video, find):
    x = 0
    while True:
        x += 1
        if find:
            if x > 1:
                return False, None, None
            try:
                check = face_cascade.detectMultiScale(cv2.cvtColor(lastface, cv2.COLOR_BGR2GRAY), 1.2, 5)
                if len(check) > 0:
                    return False, None, None
            except:
                print("ups")
        for i in range(1, 3):
            img = video.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces=[]
            faces.extend(face_cascade.detectMultiScale(gray, 1.2, 5))
        if len(faces) > 0:

            print(faces)
            odleglosc = []
            minimum = 0
            for i, (x, y, w, h) in enumerate(faces):
                odleglosc.append(math.sqrt((x - lastpos[0]) ** 2 + (y - lastpos[1]) ** 2))
                if (odleglosc[i] <= odleglosc[minimum]):
                    minimum = i
                imgCopy = img[:]
                cv2.rectangle(imgCopy, (x, y), (x + w, y + h), (0, 0, 255), 3)
                cv2.imshow('img', imgCopy)

            return True, img, faces[0]
        cv2.imshow('img', img)
        k = cv2.waitKey(1)


i = 0

ok, frame, bbox = findface(video, False)
bbox = (bbox[0], bbox[1], bbox[2], bbox[3])
ok = csrtTracker.init(frame, bbox)

while True:
    timer = cv2.getTickCount()
    i += 1
    if i >= CHECK:
        i = 0
        ok, frame, bbox = findface(video, True)
        if ok:
            bbox = (bbox[0], bbox[1], bbox[2], bbox[3])
            csrtTracker = cv2.TrackerCSRT_create()
            ok = csrtTracker.init(frame, bbox)

    frame = video.read()
    ok, bbox = csrtTracker.update(frame)

    if ok:
        p1 = (int(bbox[0]), int(bbox[1]))
        (x, y, w, h) = bbox
        print(bbox)
        lastface = frame[int(y):int(y) + int(h), int(x):int(x) + int(w)]
        lastpos = p1
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
    else:
        cv2.putText(frame, "Tracking failure detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
    cv2.putText(frame, "FPS : " + str(int(fps)), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)

    cv2.imshow("Tracking", frame)

    k = cv2.waitKey(1) & 0xff
    if k == 23:
        video.stop()
        break
