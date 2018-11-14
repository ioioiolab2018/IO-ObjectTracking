import cv2
import sys


csrtTracker = cv2.TrackerCSRT_create()
CHECK=10
video = cv2.VideoCapture("http://192.168.0.11:8080/mjpeg")

if not video.isOpened():
    sys.exit()


face_cascade = cv2.CascadeClassifier()
face_cascade.load("C:/Users/Piotr/PycharmProjects/untitled2/haarcascade_frontalface_default.xml")


def findface(video, find):
    x=0
    while True:
        x+=1
        if find :
            if x > 2:
                return False, None, None
        ret, img = video.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        if len(faces)>0:
            print(faces)
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 3)
                cv2.imshow('img', img)
            return True, img,faces[0]
        cv2.imshow('img', img)
        k = cv2.waitKey(1)
i=99999

ok, frame , bbox = findface(video, False)
bbox=(bbox[0],bbox[1],bbox[2],bbox[3])
ok = csrtTracker.init(frame, bbox)

while True:
    i += 1
    if i> CHECK:
        i=0
        ok, frame, bbox = findface(video,True)
        if ok:
            bbox = (bbox[0], bbox[1], bbox[2], bbox[3])
            csrtTracker = cv2.TrackerCSRT_create()
            ok = csrtTracker.init(frame, bbox)


    ok, frame = video.read()
    if not ok:
        print("Brak obrazu")
        continue

    timer = cv2.getTickCount()

    ok, bbox = csrtTracker.update(frame)

    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);

    if ok:

        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
    else:
        cv2.putText(frame, "Tracking failure detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

    cv2.putText(frame, "FPS : " + str(int(fps)), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)

    cv2.imshow("Tracking", frame)

    k = cv2.waitKey(1) & 0xff
    if k == 23: break