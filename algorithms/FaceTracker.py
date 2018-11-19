import math
import sys
import cv2
import os


FRAME_QUANTITY = 20


def face_run(cam = 0):
    video = capture(cam)
    face_cascade = initialize_classifier()
    run_algorithm(video, face_cascade)


def capture(cam =0):
    video = cv2.VideoCapture(cam)

    if not video.isOpened():
        print("Video can not be captured!")
        sys.exit()

    return video


def initialize_classifier():
    face_cascade = cv2.CascadeClassifier()
    face_cascade.load(os.path.abspath("algorithms/classifier/haarcascade_frontalface_default.xml"))
    # face_cascade.load(os.path.abspath("classifier/haarcascade_frontalface_default.xml"))
    return face_cascade


def run_algorithm(video, face_cascade):
    last_position = [0, 0]
    last_face = []
    counter = FRAME_QUANTITY

    csrt_tracker = cv2.TrackerCSRT_create()
    ok, frame, bbox = find_face(video, face_cascade, last_face, last_position, False)
    bbox = (bbox[0], bbox[1], bbox[2], bbox[3])
    _ = csrt_tracker.init(frame, bbox)

    while True:
        timer = cv2.getTickCount()
        counter += 1
        if counter > FRAME_QUANTITY:
            counter = 0
            ok, frame, bbox = find_face(video, face_cascade, last_face, last_position, True)
            if ok:
                bbox = (bbox[0], bbox[1], bbox[2], bbox[3])
                csrt_tracker = cv2.TrackerCSRT_create()
                _ = csrt_tracker.init(frame, bbox)

        _, frame = video.read()
        ok, bbox = csrt_tracker.update(frame)
        if ok:
            p1 = (int(bbox[0]), int(bbox[1]))
            (x, y, w, h) = bbox
            print(bbox)
            last_face = frame[int(y):int(y) + int(h), int(x):int(x) + int(w)]
            last_position = p1
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
        else:
            cv2.putText(frame, "Tracking failure detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

        draw_fps_counter(frame, timer)
        cv2.imshow("Tracking", frame)

        k = cv2.waitKey(1) & 0xff
        if k == 27:
            break


def find_face(video, face_cascade, last_face, last_position, search):
    faces = []
    counter = 0

    while True:
        counter += 1
        if search:
            if counter > 1:
                return False, None, None
            try:
                check = face_cascade.detectMultiScale(cv2.cvtColor(last_face, cv2.COLOR_BGR2GRAY), 1.2, 5)
                if len(check) > 0:
                    return False, None, None
            except:
                print("ups")

        _, img = video.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces.extend(face_cascade.detectMultiScale(gray, 1.2, 5))

        if len(faces) > 0:
            print(faces)
            distance = []
            minimum = 0
            for i, (x, y, w, h) in enumerate(faces):
                distance.append(math.sqrt((x - last_position[0]) ** 2 + (y - last_position[1]) ** 2))
                if distance[i] <= distance[minimum]:
                    minimum = i
                img_copy = img[:]
                cv2.rectangle(img_copy, (x, y), (x + w, y + h), (0, 0, 255), 3)
                cv2.imshow('img', img_copy)

            return True, img, faces[0]

        cv2.imshow('img', img)
        _ = cv2.waitKey(1)


def draw_fps_counter(frame, timer):
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
    cv2.putText(frame, "FPS : " + str(int(fps)), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)


# if __name__ == '__main__':
#     face_run()
