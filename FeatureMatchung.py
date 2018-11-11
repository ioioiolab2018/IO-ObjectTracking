import numpy as np
import cv2
from matplotlib import pyplot as plt

MIN_MATCH_COUNT = 5


video = cv2.VideoCapture("http://192.168.0.11:4747/mjpegfeed")
ok, frame = video.read()
bbox = cv2.selectROI('Tracker', frame)
(x, y, w, h) = bbox
img1 = frame[y:y + h, x:x + w]

akaze= cv2.ORB_create(nfeatures=15000,nlevels=10)
kpts1, desc1 = akaze.detectAndCompute(img1, None)




while True:

    ok, img2 = video.read()
    img4=img2[:]
    cv2.imshow('d111f', img1)


    try:
        kpts2, desc2 = akaze.detectAndCompute(img4, None)
        img2 = cv2.drawKeypoints(img1, kpts1, None, color=(0,255,0))
        matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_BRUTEFORCE_HAMMING)
        # BFMatcher with default params
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(desc1, desc2, k=2)
        # store all the good matches as per Lowe's ratio test.
        good = []
        for m,n in matches:
            if m.distance < 0.75*n.distance:
                good.append(m)

        if len(good)>MIN_MATCH_COUNT:
            src_pts = np.float32([ kpts1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
            dst_pts = np.float32([ kpts2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
            matchesMask = mask.ravel().tolist()

            h,v,x=img2.shape
            pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
            dst = cv2.perspectiveTransform(pts,M)

        else:
            print( "Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT))
            matchesMask = None

        draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                           singlePointColor = None,
                           flags = 2)

        img3 = cv2.drawMatches(img1,kpts1,img4,kpts2,good,None,**draw_params)

        cv2.imshow('df', img3)
        cv2.waitKey(1)
    except :
        print("Błąd")