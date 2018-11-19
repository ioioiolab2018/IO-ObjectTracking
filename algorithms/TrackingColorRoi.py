from random import randint
import cv2
import sys
import imutils
import numpy as np

diffA = 0.7
diffB = 1.3


def color_run(cam=0):
    video, colors, lower, upper = capture(cam)
    run_algorithm(video, colors, lower, upper)


def capture(cam = 0):
    video = cv2.VideoCapture(cam)

    if not video.isOpened():
        print("Video can not be captured!")
        sys.exit()

    lower = []
    upper = []
    colors = []
    ok, image = video.read()

    # Wczytywanie obiektów do śledzenia
    while True:
        select_object(image, colors, lower, upper)

        print("Press q to quit selecting boxes and start tracking")
        print("Press any other key to select next object")

        k = cv2.waitKey(0) & 0xFF
        if k == 113:
            break
    return video, colors, lower, upper


def select_object(image, colors, lower, upper):
    selected_object = cv2.selectROI('Tracker', image)
    (x, y, w, h) = selected_object
    roi = image[y:y + h, x:x + w]

    if roi.size > 0:
        hsv_roi = (cv2.cvtColor(roi, cv2.COLOR_BGR2HSV))
        lower.append(np.array([np.percentile(hsv_roi[:, :, 0], 1), np.percentile(hsv_roi[:, :, 1], 1),
                               np.percentile(hsv_roi[:, :, 2], 1)]) * diffA)
        upper.append(np.array([np.percentile(hsv_roi[:, :, 0], 99), np.percentile(hsv_roi[:, :, 1], 99),
                               np.percentile(hsv_roi[:, :, 2], 99)]) * diffB)
        colors.append((randint(0, 255), randint(0, 255), randint(0, 255)))


def run_algorithm(video, colors, lower, upper):
    while True:
        timer = cv2.getTickCount()
        ok, frame = video.read()

        if ok:
            blur_frame = cv2.GaussianBlur(frame, (15, 15), 0)
            hsv = cv2.cvtColor(blur_frame, cv2.COLOR_BGR2HSV)

            for i, lowerI in enumerate(lower):
                mask = cv2.inRange(hsv, lowerI, upper[i])
                mask = cv2.erode(mask, None, iterations=3)
                mask = cv2.dilate(mask, None, iterations=3)
                cv2.imshow("MASK: " + str(i), mask)
                contours = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
                contours = contours[0] if imutils.is_cv2() else contours[1]

                if len(contours) > 0:
                    max_contour = max(contours, key=cv2.contourArea)
                    rect = cv2.minAreaRect(max_contour)
                    box = cv2.boxPoints(rect)
                    box = np.int0(box)
                    cv2.drawContours(frame, [box], 0, colors[i], 2)
                    M = cv2.moments(max_contour)
                    center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                    cv2.circle(frame, center, 5, (0, 0, 255), -1)

            fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
            cv2.putText(frame, "FPS : " + str(int(fps)), (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1)
            cv2.imshow("Frame", frame)
        else:
            print("Video can not be captured!")

        k = cv2.waitKey(1) & 0xff
        if k == 27:
            break


# if __name__ == '__main__':
#     color_run()
