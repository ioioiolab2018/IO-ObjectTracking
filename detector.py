from random import randint

import numpy as np
import cv2
import imutils


def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype="float32")

    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    # return the ordered coordinates
    return rect


def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    # return the warped image
    return warped


video = cv2.VideoCapture("http://192.168.1.11:8080/mjpeg")
_, image = video.read()
bbox = cv2.selectROI('Tracker', image)
(x, y, w, h) = bbox
roi = image[y:y + h, x:x + w]
lower = []
upper = []
color = []
# Wyznaczenie koloru śledzonego obiektu
if roi.size > 0:
    hsvRoi = (cv2.cvtColor(roi, cv2.COLOR_BGR2HSV))
    lower = np.array([np.percentile(hsvRoi[:, :, 0], 1), np.percentile(hsvRoi[:, :, 1], 1),
                      np.percentile(hsvRoi[:, :, 2], 1)]) * 0.7
    upper = np.array([np.percentile(hsvRoi[:, :, 0], 99), np.percentile(hsvRoi[:, :, 1], 99),
                      np.percentile(hsvRoi[:, :, 2], 99)]) * 1.3
    color = (randint(0, 255), randint(0, 255), randint(0, 255))

while True:
    _, image = video.read()
    orig = image.copy()

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    edged = cv2.Canny(gray, 20, 250)

    #cv2.imshow("Image", image)
    #cv2.imshow("Edged", edged)
    cv2.waitKey(1)

    cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

    screenCnt = []
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            screenCnt = approx
            break

    if (len(screenCnt) == 0):
        continue

    # show the contour (outline) of the piece of paper
    cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
    cv2.imshow("Outline", image)
    cv2.waitKey(1)

    warped = four_point_transform(orig, screenCnt.reshape(4, 2))

    blurFrame = cv2.GaussianBlur(warped, (15, 15), 0)
    hsv = cv2.cvtColor(blurFrame, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.erode(mask, None, iterations=3)
    mask = cv2.dilate(mask, None, iterations=3)
    #cv2.imshow("MASK: ", mask)
    contours = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
    contours = contours[0] if imutils.is_cv2() else contours[1]
    center = None
    if len(contours) > 0:
        maxContour = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(maxContour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(warped, [box], 0, color, 2)
        M = cv2.moments(maxContour)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        cv2.circle(warped, center, 5, (0, 0, 255), -1)
        print("X: " + str(2*((center[0] / warped.shape[1])-0.5)) + " Y: " + str(2*((center[1] / warped.shape[0])-0.5)))
   # cv2.imshow("Frame", warped)

    # show the original and scanned images
    #cv2.imshow("Original", orig)
    cv2.imshow("Scanned", warped)
    cv2.waitKey(1)
