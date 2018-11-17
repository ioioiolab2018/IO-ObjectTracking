from random import randint
import cv2
import sys

# from tracking import createTrackerByName
CAMERA_ADDRESS = 0  # "http://192.168.0.11:8080/mjpeg"


def csrt_run():
    video, image, colors, selected_objects = capture()
    multi_tracker = initialize_tracker(image, selected_objects)
    run_algorithm(video, multi_tracker, colors)


def capture():
    video = cv2.VideoCapture(CAMERA_ADDRESS)

    if not video.isOpened():
        print("Video can not be captured!")
        sys.exit()

    selected_objects = []
    colors = []
    ok, image = video.read()

    # Wczytywanie obiektów do śledzenia
    while True:
        select_object(image, colors, selected_objects)

        print("Press q to quit selecting boxes and start tracking")
        print("Press any other key to select next object")

        k = cv2.waitKey(0) & 0xFF
        if k == 113:
            break
    return video, image, colors, selected_objects


def select_object(image, colors, selected_objects):
    selected_object = cv2.selectROI('Tracker', image)
    (x, y, w, h) = selected_object
    roi = image[y:y + h, x:x + w]

    if roi.size > 0:
        selected_objects.append(selected_object)
        colors.append((randint(0, 255), randint(0, 255), randint(0, 255)))


def initialize_tracker(image, selected_objects):
    multi_tracker = cv2.MultiTracker_create()
    for bbox in selected_objects:
        multi_tracker.add(cv2.TrackerCSRT_create(), image, bbox)
        # multiTracker.add(createTrackerByName("CSRT"), image, bbox)
    return multi_tracker


def run_algorithm(video, multi_tracker, colors):
    while True:
        ok, frame = video.read()

        if ok:
            timer = cv2.getTickCount()
            ok, boxes = multi_tracker.update(frame)
            fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

            # Narysuj wynik
            if ok:
                for i, new_box in enumerate(boxes):
                    p1 = (int(new_box[0]), int(new_box[1]))
                    p2 = (int(new_box[0] + new_box[2]), int(new_box[1] + new_box[3]))
                    cv2.rectangle(frame, p1, p2, colors[i], 2, 1)
            else:
                cv2.putText(frame, "Tracking error", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
            cv2.putText(frame, "FPS : " + str(int(fps)), (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1)
            cv2.imshow("Tracking", frame)
        else:
            print("Video can not be captured!")

        k = cv2.waitKey(1) & 0xff
        if k == 27:
            break


# if __name__ == '__main__':
#     csrt_run()
